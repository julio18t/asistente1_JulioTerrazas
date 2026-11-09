"""Pruebas sin modelo real ni internet: uv run pytest

Un modelo falso (FunctionModel) pide la tool que queremos, y comprobamos lo que hace el código.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY_LOCAL", "clave-de-prueba")  # para importar sin .env (no se llama a Gemini)

import httpx  # noqa: E402
import pytest  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

import api  # noqa: E402
import asistente  # noqa: E402
import bot  # noqa: E402
import capacidades  # noqa: E402
from config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    """Cada prueba usa su propia gastos.db, no la de datos/."""
    monkeypatch.setattr(capacidades, "DB", tmp_path / "gastos.db")


def pide(tool: str, args: dict) -> FunctionModel:
    """Un modelo falso: la 1.ª vez pide la tool; cuando recibe su resultado, responde texto."""

    def responder(mensajes, info):
        ultimo = mensajes[-1].parts[-1]
        if isinstance(ultimo, ToolReturnPart):
            return ModelResponse(parts=[TextPart(f"Resultado: {ultimo.content}")])
        return ModelResponse(parts=[ToolCallPart(tool, args)])

    return FunctionModel(responder)


def test_consultar_gastos_sin_modelo():
    gastos = capacidades.consultar_gastos("transporte")  # una tool es una función normal
    assert gastos["total_bs"] == 67


def test_registrar_gasto_escribe_en_la_base():
    respuesta = capacidades.registrar_gasto(
        fecha="2030-05-14",
        categoria="ocio",
        descripcion="Teatro",
        monto=30,
    )

    assert "Gasto registrado" in respuesta
    assert capacidades.consultar_gastos("ocio")["total_bs"] == 70


def test_monto_negativo_no_se_guarda():
    respuesta = capacidades.registrar_gasto(
        fecha="2030-05-14",
        categoria="ocio",
        descripcion="Error",
        monto=-5,
    )

    assert "No se guardó" in respuesta
    assert capacidades.consultar_gastos("ocio")["total_bs"] == 40


async def test_chat_ajeno_se_rechaza_sin_llamar_al_modelo(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", "123")
    respuestas = []

    async def reply_text(texto):
        respuestas.append(texto)

    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999), message=SimpleNamespace(text="hola", reply_text=reply_text)
    )
    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        await bot.mensaje(update, None)
    assert "No autorizado" in respuestas[0] and "999" in respuestas[0]


def test_permitidos(monkeypatch):
    monkeypatch.setattr(settings, "telegram_permitidos", " 123, 456 ,")
    assert settings.permitidos == {"123", "456"}
    monkeypatch.setattr(settings, "telegram_permitidos", "")
    assert settings.permitidos == set()  # vacío = nadie


# ── Extra opcional: la API (API.md) ──


async def post_chat(cuerpo: dict, token: str | None = None) -> httpx.Response:
    """Llama a la API en el mismo proceso, sin servidor ni red."""
    cabeceras = {"Authorization": f"Bearer {token}"} if token else {}
    transporte = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
        return await cliente.post("/chat", json=cuerpo, headers=cabeceras)


async def test_api_sin_token_valido_no_llama_al_modelo(monkeypatch):
    def modelo_prohibido(mensajes, info):
        raise AssertionError("no debía llamarse al modelo")

    with asistente.asistente.override(model=FunctionModel(modelo_prohibido)):
        monkeypatch.setattr(settings, "api_tokens", SecretStr(""))
        assert (await post_chat({"mensaje": "hola"}, "cualquiera")).status_code == 401  # vacío = nadie
        monkeypatch.setattr(settings, "api_tokens", SecretStr("abc, def"))
        assert (await post_chat({"mensaje": "hola"})).status_code == 401
        assert (await post_chat({"mensaje": "hola"}, "xyz")).status_code == 401
        assert (await post_chat({"mensaje": ""}, "abc")).status_code == 422  # Pydantic valida el pedido


async def test_api_con_token_responde(monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", SecretStr("abc,def"))
    monkeypatch.setattr(
        capacidades,
        "buscar",
        lambda pregunta, k=3: [
            {
                "fuente": "paper_prueba-0",
                "similitud": 0.9,
                "texto": "El estudio utiliza simulación Monte Carlo.",
            }
        ],
    )

    with asistente.asistente.override(
        model=pide(
            "buscar_documentos",
            {"pregunta": "¿Qué método utiliza el estudio?"},
        )
    ):
        r = await post_chat(
            {"mensaje": "¿Qué método utiliza el estudio?"},
            "def",
        )

    assert r.status_code == 200
    assert "Monte Carlo" in r.json()["texto"]
    assert "paper_prueba-0" in r.json()["texto"]


async def test_asistente_consulta_papers(monkeypatch):
    consultas = []

    def buscar_falso(pregunta, k=3):
        consultas.append(pregunta)
        return [
            {
                "fuente": "paper_prueba-0",
                "similitud": 0.9,
                "texto": "El estudio utiliza simulación Monte Carlo.",
            }
        ]

    monkeypatch.setattr(capacidades, "buscar", buscar_falso)

    with asistente.asistente.override(
        model=pide(
            "buscar_documentos",
            {"pregunta": "¿Qué método utiliza el estudio?"},
        )
    ):
        respuesta = await asistente.responder(
            "¿Qué método utiliza el estudio?"
        )

    assert consultas == ["¿Qué método utiliza el estudio?"]
    assert "Monte Carlo" in respuesta
    assert "paper_prueba-0" in respuesta


def test_buscar_papers_openalex(monkeypatch):
    def get_falso(url, params, timeout, headers):
        assert url == "https://api.openalex.org/works"
        assert params["search"] == "farm simulation"
        assert params["per_page"] == 3

        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "results": [{
                    "display_name": "Artículo ficticio de prueba",
                    "publication_year": 2024,
                    "doi": None,
                    "id": "https://openalex.org/W000",
                    "authorships": [
                        {"author": {"display_name": "Autor de prueba"}}
                    ],
                    "open_access": None,
                }]
            },
        )

    monkeypatch.setattr(capacidades.httpx, "get", get_falso)

    resultado = capacidades.buscar_papers("farm simulation", 3)
    publicacion = resultado["publicaciones"][0]

    assert resultado["fuente"] == "OpenAlex"
    assert publicacion["autores"] == ["Autor de prueba"]
    assert publicacion["doi"] is None
    assert publicacion["enlace_acceso_abierto"] is None