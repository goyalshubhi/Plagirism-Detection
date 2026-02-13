from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.manager import BaseUserManager, UUIDIDMixin
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from uuid import UUID

from app.core.config import settings
from app.core.db import get_db
from app.models.user import User
from app.schemas import UserCreate, UserRead, UserUpdate

# ---------------- JWT STRATEGY ----------------
def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=settings.SECRET_KEY, lifetime_seconds=3600)

# ---------------- TRANSPORTS ----------------
# Bearer is best for Swagger testing
bearer_transport = BearerTransport(tokenUrl="/api/auth/jwt/login")

# Cookie is useful for browser sessions
cookie_transport = CookieTransport(
    cookie_name="plagiarism_auth",
    cookie_max_age=3600,
    cookie_httponly=True,
    cookie_secure=False,  # IMPORTANT for localhost HTTP
)

# ---------------- BACKENDS ----------------
auth_backend_bearer = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

auth_backend_cookie = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# ---------------- USER MANAGER ----------------
class UserManager(UUIDIDMixin, BaseUserManager[User, UUID]):
    reset_password_token_secret = settings.SECRET_KEY
    verification_token_secret = settings.SECRET_KEY

    async def on_after_register(self, user: User, request=None):
        print(f"User {user.id} registered.")

async def get_user_db(session: AsyncSession = Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

# ---------------- FASTAPI USERS ----------------
fastapi_users = FastAPIUsers[User, UUID](
    get_user_manager,
    [auth_backend_bearer, auth_backend_cookie],
)

current_user = fastapi_users.current_user(active=True)

# ---------------- ROUTER EXPORT (THIS IS WHAT SWAGGER NEEDS) ----------------
router = APIRouter()

# Auth routes
router.include_router(
    fastapi_users.get_auth_router(auth_backend_bearer),
    prefix="/jwt",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_auth_router(auth_backend_cookie),
    prefix="/cookie",
    tags=["auth"],
)

# Register + users
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)
