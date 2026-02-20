from rdflib import Namespace

HTTPOnt = Namespace("https://www.w3.org/2011/http#")

Message = HTTPOnt["Message"]

Request = HTTPOnt["Request"]

methodName = HTTPOnt["methodName"]

requestURI = HTTPOnt["requestURI"]

body = HTTPOnt["body"]
