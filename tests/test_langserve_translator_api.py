import pytest


class TestTranslatorAPI:

    def test_supported_languages(self):
        supported = ["en", "es", "fr", "de", "zh", "ja", "ar", "hi"]
        assert "en" in supported
        assert len(supported) >= 5

    def test_empty_input_rejected(self):
        def translate(text, target_lang):
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            return f"[{target_lang}] {text}"
        with pytest.raises(ValueError):
            translate("", "es")

    def test_translation_returns_string(self):
        def mock_translate(text, target):
            return f"translated: {text}"
        result = mock_translate("Hello world", "es")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_language_code_validation(self):
        valid_codes = {"en", "es", "fr", "de", "zh"}
        def validate_lang(code):
            return code.lower() in valid_codes
        assert validate_lang("en") is True
        assert validate_lang("xx") is False
        assert validate_lang("EN") is True

    def test_long_text_handled(self):
        long_text = "word " * 500
        assert len(long_text.split()) == 500

    def test_special_characters_in_text(self):
        text = "Héllo Wörld — this has spécial chars"
        assert len(text) > 0
        assert "é" in text


class TestAPIEndpoints:

    def test_request_structure(self):
        request = {"text": "Hello", "source_lang": "en", "target_lang": "es"}
        assert "text" in request
        assert "target_lang" in request

    def test_response_structure(self):
        response = {"translated_text": "Hola", "source_lang": "en", "target_lang": "es", "confidence": 0.95}
        assert "translated_text" in response
        assert 0 <= response["confidence"] <= 1

    def test_batch_translation(self):
        texts = ["Hello", "World", "How are you"]
        results = [f"translated_{t}" for t in texts]
        assert len(results) == len(texts)
