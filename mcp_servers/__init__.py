"""Tầng MCP (Model Context Protocol) §V — đóng gói năng lực truy xuất/đồ thị/thẩm định
qua giao diện định kiểu chặt (Pydantic). Logic thuần tách khỏi wrapper FastMCP để test
được mà không cần cài thư viện `mcp`; wrapper chỉ nạp khi chạy server thật."""
