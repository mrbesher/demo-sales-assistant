import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from calculations import calculate


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=1000)


class OpenInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ids: list[str] = Field(min_length=1, max_length=10)


class CalculateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expression: str = Field(min_length=1, max_length=300)
    units: str = Field(min_length=1, max_length=80)
    source_ids: list[str] = Field(min_length=1, max_length=10)
    assumptions: str = Field(max_length=1500)


class ProcessInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    measurements: dict[str, float]
    filters: dict[str, list[float]] | None = None


INPUTS = {
    "search_brochures": SearchInput,
    "open_chunks": OpenInput,
    "search_service_reports": SearchInput,
    "calculate": CalculateInput,
    "analyze_process": ProcessInput,
}

DESCRIPTIONS = {
    "search_brochures": "Search product evidence. Returns opened best match with page images, "
    "same-page chunks and up to ten candidate descriptions. Initial search plus three more per turn.",
    "open_chunks": "Open listed chunk IDs with their page images. Descriptions are navigation, "
    "not evidence. Open candidates before making claims from them.",
    "search_service_reports": "Search historical visits for comparable work or local pricing. "
    "Returns three whole reports and extracted fields. Search task/equipment for effort, then "
    "search country/site and labour or travel rates separately for local pricing. A rate search "
    "need not repeat the task or equipment. Broaden queries if the same reports return.",
    "calculate": "Compute arithmetic. Identify sources, result units and transfer assumptions. "
    "Never mix currencies. Source IDs must refer to opened evidence or prior calculations.",
    "analyze_process": "Compare customer readings with historical plant data. Accept exact column "
    "names as measurement keys and optional inclusive [minimum, maximum] filters. "
    "Returns distribution statistics, not causal operating advice. Available columns: ",
}


class Evidence:
    def __init__(self, brochures, reports, process, storage: Path, model):
        self.brochures = brochures
        self.reports = reports
        self.process = process
        self.storage = storage
        self.model = model
        self.records = {}
        for record in brochures.records + reports.records:
            self.records[record["id"]] = record
        self.tools = []
        for name, schema in INPUTS.items():
            description = DESCRIPTIONS[name]
            if name == "analyze_process":
                columns = ", ".join(process.columns)
                description += columns
            parameters = schema.model_json_schema()
            function = {"name": name, "description": description, "parameters": parameters}
            self.tools.append({"type": "function", "function": function})

    def open(self, ids: list[str], state: dict) -> tuple[list[dict], list[dict]]:
        requested = []
        attached = set(state["pages"])
        for source_id in ids:
            known = source_id in state["candidates"] or source_id in state["sources"]
            if not known or source_id not in self.records:
                raise ValueError("Only chunk IDs returned by search can be opened.")
            record = self.records[source_id]
            if record["kind"] != "brochure":
                raise ValueError("Only brochure chunks can be opened with this tool.")
            requested.append(record)
            attached.update(record["pages"])
        for record in self.brochures.records:
            required = set(record["pages"])
            if required and required <= attached and record not in requested:
                requested.append(record)
        opened = []
        images = []
        for record in requested:
            source_id = record["id"]
            if source_id in state["sources"]:
                continue
            page_images = zip(record["pages"], record["image_paths"], strict=True)
            for page, image_path in page_images:
                if page in state["pages"]:
                    continue
                path = self.storage / image_path
                image = self.model.image(path)
                images.append({"type": "text", "text": f"Source page: {page}"})
                images.append(image)
                state["pages"].add(page)
            state["sources"][source_id] = record
            opened.append({"id": source_id, "text": record["text"], "pages": record["pages"]})
        return opened, images

    def execute(self, name: str, arguments: dict, state: dict) -> tuple[dict, list[dict]]:
        if name not in INPUTS:
            raise ValueError("Unknown tool. Use one of the supplied tools.")
        schema = INPUTS[name]
        values = schema.model_validate(arguments)
        images = []
        if name == "search_brochures":
            if state["searches"] >= 4:
                raise ValueError(
                    "The initial brochure search and three additional searches are used."
                )
            state["searches"] += 1
            matches = self.brochures.search(values.query, limit=10)
            candidates = []
            for record in matches:
                source_id = record["id"]
                state["candidates"].add(source_id)
                candidates.append(
                    {"id": source_id, "description": record["title"], "pages": record["pages"]}
                )
            opened = []
            if matches:
                first_id = matches[0]["id"]
                opened, images = self.open([first_id], state)
            result = {
                "opened": opened,
                "candidates": candidates,
                "searches_remaining": 4 - state["searches"],
            }
        elif name == "open_chunks":
            opened, images = self.open(values.ids, state)
            result = {"opened": opened, "already_open": []}
            opened_ids = {record["id"] for record in opened}
            for source_id in values.ids:
                if source_id not in opened_ids:
                    result["already_open"].append(source_id)
        elif name == "search_service_reports":
            matches = self.reports.search(values.query, limit=3)
            reports = []
            for record in matches:
                if record["id"] in state["sources"]:
                    reports.append({"id": record["id"], "already_open": True})
                    continue
                state["sources"][record["id"]] = record
                reports.append(
                    {"id": record["id"], "fields": record["fields"], "text": record["text"]}
                )
            result = {"reports": reports}
        elif name == "calculate":
            for source_id in values.source_ids:
                if source_id not in state["sources"]:
                    raise ValueError("A calculation references evidence that has not been opened.")
            result = calculate(values.expression)
            result["units"] = values.units
            result["source_ids"] = values.source_ids
            result["assumptions"] = values.assumptions
        else:
            result = self.process.analyze(values.measurements, values.filters)
        if name in ("calculate", "analyze_process"):
            state["sequence"] += 1
            source_id = f"result-{state['sequence']}"
            result["id"] = source_id
            text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            state["sources"][source_id] = {
                "id": source_id,
                "kind": "calculation" if name == "calculate" else "process",
                "title": values.expression
                if name == "calculate"
                else "Historical plant comparison",
                "text": text,
                "pages": [],
                "image_paths": [],
                "fields": result,
            }
        return result, images
