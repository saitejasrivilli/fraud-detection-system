"""
FastAPI Authentication
JWT tokens, OAuth2, role-based access control
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import jwt
import os
from passlib.context import CryptContext

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
http_bearer = HTTPBearer()


class Token(BaseModel):
    """Token response"""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload"""
    username: str
    scopes: list = []
    role: str = "user"


class User(BaseModel):
    """User model"""
    username: str
    email: str
    role: str = "user"  # admin, reviewer, analyst, user
    active: bool = True


class UserInDB(User):
    """User with hashed password"""
    hashed_password: str


# Mock users database (replace with real database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@fraud.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "active": True
    },
    "reviewer": {
        "username": "reviewer",
        "email": "reviewer@fraud.com",
        "hashed_password": pwd_context.hash("reviewer123"),
        "role": "reviewer",
        "active": True
    },
    "analyst": {
        "username": "analyst",
        "email": "analyst@fraud.com",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
        "active": True
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user"""
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return UserInDB(**user)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """Get current user from token"""
    credential_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise credential_exception
        
        token_data = TokenData(username=username, role=payload.get("role", "user"))
        
    except jwt.InvalidTokenError:
        raise credential_exception
    
    return token_data


async def get_current_active_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Get current active user"""
    user = USERS_DB.get(current_user.username)
    if not user or not user.get("active"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(*allowed_roles: str):
    """Require specific roles"""
    async def role_checker(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker


class Auth:
    """Authentication utilities"""
    
    @staticmethod
    async def login(username: str, password: str) -> Optional[Token]:
        """Login user"""
        user = authenticate_user(username, password)
        if not user:
            return None
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    @staticmethod
    async def verify_token(token: str) -> Optional[TokenData]:
        """Verify token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            return TokenData(username=username, role=payload.get("role", "user"))
        except jwt.InvalidTokenError:
            return None
