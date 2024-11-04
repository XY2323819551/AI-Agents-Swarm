from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from tool_pool import BaseTool

class WikiDataTool(BaseTool):
    def __init__(self):
        self.tool = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
        super().__init__()

    def run(self, query):
        return self.tool.run(query)
    
if __name__ == "__main__":
    query = "Alan Turing"
    tool = WikiDataTool()
    print(tool.run(query))
