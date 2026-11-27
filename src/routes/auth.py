import uuid

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.auth import create_access_token, hash_password, verify_password
from src.logger import get_logger, set_user_id
from src.rate_limit import limiter
from src.routes.dependencies import get_db
from src.schema.models import AuthResponse, LoginRequest, SignupRequest, UserResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _create_user(email: str, password: str, db) -> AuthResponse:
    normalized_email = email.strip().lower()
    logger.debug("Signup request received")
    if not normalized_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required",
        )

    password_hash = hash_password(password)
    user_id = str(uuid.uuid4())
    set_user_id(user_id)

    async with db.acquire() as connection:
        try:
            await connection.execute(
                """
                INSERT INTO users (id, email, password_hash)
                VALUES ($1, $2, $3)
                """,
                user_id,
                normalized_email,
                password_hash,
            )
        except asyncpg.exceptions.UniqueViolationError as error:
            logger.info("Signup rejected because the email is already registered")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email is already registered",
            ) from error

    token = create_access_token(user_id=user_id, email=normalized_email)
    logger.info("User signed up successfully: user_id=%s", user_id)
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(id=user_id, email=normalized_email),
    )


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def signup(request: Request, body: SignupRequest, db=Depends(get_db)) -> AuthResponse:
    try:
        return await _create_user(body.email, body.password, db)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Signup failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to create user",
        ) from error


@router.post("/login", response_model=AuthResponse)
@limiter.limit("10/minute")
async def login(request: Request, body: LoginRequest, db=Depends(get_db)) -> AuthResponse:
    normalized_email = body.email.strip().lower()
    logger.debug("Login request received")
    try:
        async with db.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT id, email, password_hash
                FROM users
                WHERE email = $1
                """,
                normalized_email,
            )

        if row is None or not verify_password(body.password, row["password_hash"]):
            # No email here: logs must not hold personal data. The rate
            # limiter, not the logs, protects against password guessing.
            logger.warning("Login failed: unknown email or wrong password")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        user_id = str(row["id"])
        set_user_id(user_id)
        token = create_access_token(user_id=user_id, email=row["email"])
        logger.info("User logged in successfully: user_id=%s", user_id)
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse(id=user_id, email=row["email"]),
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Login failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to sign in",
        ) from error
