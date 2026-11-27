import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from src.config.settings import Settings
from src.services.provider_chain import parse_provider_chain


def make_settings(**env):
    """Settings built from *env* only, not from the real process environment."""
    with patch.dict(os.environ, env, clear=True):
        return Settings()


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = make_settings()

        self.assertEqual(settings.milvus_port, 19530)
        self.assertEqual(settings.max_upload_bytes, 25 * 1024 * 1024)
        self.assertTrue(settings.reranker_enabled)
        self.assertIsNone(settings.reranker_url)

    def test_reranker_top_k_defaults_to_milvus_top_k(self):
        self.assertEqual(make_settings(MILVUS_TOP_K="7").reranker_top_k, 7)
        self.assertEqual(
            make_settings(MILVUS_TOP_K="7", RERANKER_TOP_K="3").reranker_top_k, 3
        )

    def test_empty_variable_uses_default(self):
        self.assertEqual(make_settings(MILVUS_PORT="").milvus_port, 19530)

    def test_bool_parsing(self):
        self.assertTrue(make_settings(MINIO_SECURE="yes").minio_secure)
        self.assertFalse(make_settings(RERANKER_ENABLED="false").reranker_enabled)

    def test_invalid_value_fails_with_setting_name(self):
        with self.assertRaises(ValidationError) as error:
            make_settings(MILVUS_PORT="abc")

        self.assertIn("milvus_port", str(error.exception))

    def test_out_of_range_value_fails(self):
        with self.assertRaises(ValidationError):
            make_settings(ARQ_MAX_TRIES="0")

    def test_postgres_kwargs_name_every_missing_setting(self):
        with self.assertRaises(RuntimeError) as error:
            make_settings(PG_HOST="db").postgres_pool_kwargs()

        message = str(error.exception)
        for name in ("PG_PORT", "PG_DATABASE", "PG_USER", "PG_PASSWORD"):
            self.assertIn(name, message)
        self.assertNotIn("PG_HOST", message)

    def test_postgres_kwargs(self):
        settings = make_settings(
            PG_HOST="db", PG_PORT="5432", PG_DATABASE="app",
            PG_USER="u", PG_PASSWORD="p", DB_POOL_SIZE="4",
        )

        kwargs = settings.postgres_pool_kwargs()

        self.assertEqual(kwargs["port"], 5432)
        self.assertEqual(kwargs["password"], "p")
        self.assertEqual(kwargs["max_size"], 4)

    def test_jwt_secret_accepts_legacy_name(self):
        self.assertEqual(make_settings(AUTH_API_KEY="legacy").require_jwt_secret(), "legacy")
        with self.assertRaises(RuntimeError):
            make_settings().require_jwt_secret()

    def test_secrets_are_hidden_in_repr(self):
        settings = make_settings(OPENAI_API_KEY="sk-very-secret", PG_PASSWORD="pw-secret")

        self.assertNotIn("sk-very-secret", repr(settings))
        self.assertNotIn("pw-secret", repr(settings))
        self.assertEqual(settings.require_openai_api_key(), "sk-very-secret")

    def test_reranker_url_trailing_slash_is_removed(self):
        self.assertEqual(
            make_settings(RERANKER_URL="http://reranker:8080/").reranker_url,
            "http://reranker:8080",
        )


class ProviderChainTests(unittest.TestCase):
    def test_parses_entries_in_order(self):
        self.assertEqual(
            parse_provider_chain("OpenAI:gpt-4o-mini, openai:gpt-4o", "LLM_FALLBACK_CHAIN"),
            [("openai", "gpt-4o-mini"), ("openai", "gpt-4o")],
        )

    def test_bad_entry_names_the_setting(self):
        with self.assertRaises(RuntimeError) as error:
            parse_provider_chain("gpt-4o", "LLM_FALLBACK_CHAIN")

        self.assertIn("LLM_FALLBACK_CHAIN", str(error.exception))


if __name__ == "__main__":
    unittest.main()
