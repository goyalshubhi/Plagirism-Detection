from fastapi import APIRouter
from app.api.auth import auth_backend, fastapi_users
from app.schemas import UserRead, UserCreate

router = APIRouter()

# login/logout
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["auth"],
)

# register
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="",
    tags=["auth"],
)
