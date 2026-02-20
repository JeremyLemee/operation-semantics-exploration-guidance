from abc import ABC, abstractmethod
import os
from typing import Any, Optional

from flask import Flask, request, Response, abort
from rdflib import Graph

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGIES_DIR = os.path.join(BASE_DIR, "ontologies")


def _determine_ontology_serialization(accept_header: Optional[str]):
    if not accept_header:
        accept_header = ""
    if "application/ld+json" in accept_header:
        return "json-ld", "application/ld+json"
    if "application/json" in accept_header:
        return "json-ld", "application/json"
    if "text/turtle" in accept_header:
        return "turtle", "text/turtle"
    return "turtle", "text/turtle"


def _serve_ontology(name: str):
    ontology_path = os.path.join(ONTOLOGIES_DIR, f"{name}.ttl")
    if not os.path.isfile(ontology_path):
        abort(404, description=f"Ontology '{name}' not found")
    graph = Graph()
    graph.parse(ontology_path, format="turtle")
    format_name, content_type = _determine_ontology_serialization(
        request.headers.get("Accept")
    )
    return Response(graph.serialize(format=format_name), content_type=content_type)


class Model(ABC):
    def __init__(self, base_url: str, **kwargs: Any):
        self._base_url = base_url
        self._info: dict[str, Graph] = {}
        self._policy = "all"
        self.initialize(**kwargs)

    @property
    def info(self) -> dict[str, Graph]:
        return self._info

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def process(self, req: Any, response: Response) -> dict[str, str]:
        pass

    def set_policy(self, policy: str) -> None:
        self._policy = policy

    def get_policy(self) -> str:
        return self._policy


class ExplorationGuidance:
    def __init__(self, model: Model, app: Flask):

        self._model = model
        self.app = app
        self.setup()

    def setup(self):
        def _render_info_model(info_model: Graph) -> Response:
            accept_header = request.headers.get("Accept") or ""
            if "text/turtle" in accept_header:
                return Response(info_model.serialize(format="turtle"), content_type="text/turtle")
            if "application/ld+json" in accept_header:
                return Response(
                    info_model.serialize(format="json-ld"),
                    content_type="application/ld+json",
                )
            if "application/json" in accept_header:
                return Response(
                    info_model.serialize(format="json-ld"),
                    content_type="application/json",
                )
            return Response(
                info_model.serialize(format="json-ld"),
                content_type="application/ld+json",
            )

        @self.app.route("/ontologies/<name>")
        def ontologies(name):
            return _serve_ontology(name)

        @self.app.route("/exploration_guidance_info/<name>")
        def find(name):
            if name not in self._model.info:
                abort(404, description=f"Mock '{name}' not found")
            info_model = self._model.info[name]
            return _render_info_model(info_model)

        @self.app.route("/policy", methods=["GET", "POST"])
        def policy():
            if request.method == "GET":
                return {"policy": self._model.get_policy()}
            data = request.get_json(silent=True)
            payload = data if isinstance(data, dict) else {}
            policy_value = request.args.get("policy") or payload.get("policy")
            if not isinstance(policy_value, str) or not policy_value:
                abort(400, description="Provide a non-empty policy value.")
            try:
                self._model.set_policy(policy_value)
            except ValueError as exc:
                abort(400, description=str(exc))
            return {"policy": self._model.get_policy()}

        @self.app.after_request
        def apply_transformation(response: Response):
            if request.path == "/policy":
                return response
            r = self._model.process(request, response)
            response.headers["Link"] = "<" + r["link"] + '>;rel="guidance"'
            return response

    def run(self, **kwargs: Any) -> None:
        self.app.run(**kwargs)


class ExplorationGuidanceProxy:
    def __init__(self, base_url: str, model: Model):
        self.base_url = base_url
        self._model = model
        self.app = Flask(__name__)
        self.setup()

    def setup(self):
        def _render_info_model(info_model: Graph) -> Response:
            accept_header = request.headers.get("Accept") or ""
            if "text/turtle" in accept_header:
                return Response(info_model.serialize(format="turtle"), content_type="text/turtle")
            if "application/ld+json" in accept_header:
                return Response(
                    info_model.serialize(format="json-ld"),
                    content_type="application/ld+json",
                )
            if "application/json" in accept_header:
                return Response(
                    info_model.serialize(format="json-ld"),
                    content_type="application/json",
                )
            return Response(
                info_model.serialize(format="json-ld"),
                content_type="application/ld+json",
            )

        @self.app.route("/ontologies/<name>")
        def ontologies(name):
            return _serve_ontology(name)

        @self.app.route("/exploration_guidance_info/<name>")
        def find(name):
            if name not in self._model.info:
                abort(404, description=f"Mock '{name}' not found")
            info_model = self._model.info[name]
            return _render_info_model(info_model)

        @self.app.route("/policy", methods=["GET", "POST"])
        def policy():
            if request.method == "GET":
                return {"policy": self._model.get_policy()}
            data = request.get_json(silent=True)
            payload = data if isinstance(data, dict) else {}
            policy_value = request.args.get("policy") or payload.get("policy")
            if not isinstance(policy_value, str) or not policy_value:
                abort(400, description="Provide a non-empty policy value.")
            try:
                self._model.set_policy(policy_value)
            except ValueError as exc:
                abort(400, description=str(exc))
            return {"policy": self._model.get_policy()}

        @self.app.route(
            "/<path:rest>",
            methods=[
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
                "OPTIONS",
                "HEAD",
                "CONNECT",
                "TRACE",
            ],
        )
        def proxy_route(rest):
            target_url = f"{self.base_url}/{rest}"
            response = requests.request(
                method=request.method,
                url=target_url,
                headers={
                    key: value for (key, value) in request.headers if key != "Host"
                },
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False,
            )
            return Response(
                response.content, status=response.status_code, headers=response.headers
            )

        @self.app.after_request
        def apply_transformation(response: Response):
            if request.path == "/policy":
                return response
            r = self._model.process(request, response)
            response.headers["Link"] = "<" + r["link"] + '>;rel="guidance"'
            return response
