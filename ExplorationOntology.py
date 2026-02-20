from rdflib import Namespace

Guidance = Namespace("http://localhost:5001/ontologies/guidance.ttl/")

Operation = Guidance["Operation"]

ExplorableOperation = Guidance["ExplorableOperation"]

NonExplorableOperation = Guidance["NonExplorableOperation"]

hasDangerCause = Guidance["hasDangerCause"]

hasOutcome = Guidance["hasOutcome"]
