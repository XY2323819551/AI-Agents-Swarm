from langchain_community.utilities import TextRequestsWrapper
from tool_pool import BaseTool

class RequestUrlTool(BaseTool):
    def __init__(self):
        self.tool = TextRequestsWrapper()
        super().__init__()
        
    def run(self, url):
        return self.tool.get(url)
    
if __name__ == "__main__":
    tool = RequestUrlTool()
    print(tool.run("https://www.google.com"))