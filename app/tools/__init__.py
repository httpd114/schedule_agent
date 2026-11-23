"""工具包：导入即完成插件注册。"""
from app.tools.calendar_tools import (  # noqa: F401
    CheckRoomOccupancyTool,
    CreateScheduleTool,
    DeleteScheduleTool,
    QuerySchedulesTool,
)
from app.tools.doc_tools import SearchDocumentsTool  # noqa: F401
from app.tools.leave_tools import ApplyLeaveTool  # noqa: F401
from app.tools.todo_tools import CreateTodoTool, QueryTodosTool  # noqa: F401


def load_tools() -> None:
    """显式触发注册（import 本模块即可，此函数供入口处可读性使用）。"""
