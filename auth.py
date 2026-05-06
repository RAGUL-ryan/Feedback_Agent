"""
auth.py  (place in project root)
─────────────────────────────────────────────────────────────────────────────
Authentication routes and JWT helpers.

Routes:
  POST /auth/register  — create account
  POST /auth/login     — get JWT token
  GET  /auth/me        — get current user details (requires token)

JWT flow:
  1. User logs in  → server returns a token (valid 7 days)
  2. Frontend stores token in localStorage
  3. Every request to /analyze sends: Authorization: Bearer <token>
  4. Backend decodes token → knows who the user is → no DB query needed
"""

import os
from datetime import datetime, timedelta
import hashlib
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from database import User, get_db

router   = APIRouter(prefix="/auth", tags=["auth"])
bearer   = HTTPBearer()
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── JWT config ─────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("JWT_SECRET_KEY", "change-this-in-production-use-a-long-random-string")
ALGORITHM   = "HS256"
TOKEN_DAYS  = 7


# ── Pydantic schemas ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    phone: str | None = None
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    full_name: str
    email: str
    phone: str | None

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ── Helpers ─────────────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    # Convert to fixed length before bcrypt
    pwd = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_ctx.hash(pwd)


def _verify(plain: str, hashed: str) -> bool:
    pwd = hashlib.sha256(plain.encode("utf-8")).hexdigest()
    return pwd_ctx.verify(pwd, hashed)

def _create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(days=TOKEN_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency — decodes the Bearer token and returns the User object.
    Used in protected routes like /analyze.
    """
    token = credentials.credentials
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id  = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
        )
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


# ── Routes ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenOut, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new account. Returns a JWT token immediately — no separate login needed."""
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists.",
        )
    user = User(
        full_name       = req.full_name.strip(),
        email           = req.email.lower().strip(),
        phone           = req.phone,
        hashed_password = _hash(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = _create_token(user.id)
    return TokenOut(access_token=token, user=UserOut.from_orm(user))


@router.post("/login", response_model=TokenOut)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Log in with email + password. Returns a JWT token."""
    user = db.query(User).filter(User.email == req.email.lower().strip()).first()
    if not user or not _verify(req.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
        )
    token = _create_token(user.id)
    return TokenOut(access_token=token, user=UserOut.from_orm(user))


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    """Return the currently logged-in user's details."""
    return current_user