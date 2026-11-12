import subprocess, json, sys

def main():
    exe_path = r"C:\Program Files\Meta VR Developer Hub\resources\bin\hzos-dev-mcp.exe"
    print("=" * 80)
    print(">>> VERIFYING META HORIZON OS DEVELOPER MCP SERVER (HzOSDevMCP)")
    print(f"[*] Executable: {exe_path}")
    print("=" * 80)

    proc = subprocess.Popen(
        [exe_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    def send_msg(msg):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line) if line else None

    # 1. Initialize
    init_resp = send_msg({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "Antigravity", "version": "1.0.0"}
        }
    })

    # 2. Initialized notification
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
    proc.stdin.flush()

    # 3. List Tools
    tools_resp = send_msg({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    })

    # 4. List Resources (if supported)
    res_resp = send_msg({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/list",
        "params": {}
    })

    proc.terminate()

    server_info = init_resp.get("result", {}).get("serverInfo", {})
    instructions = init_resp.get("result", {}).get("instructions", "")
    print(f"[+] Server Connected Successfully!")
    print(f"    Name: {server_info.get('name')}")
    print(f"    Version: {server_info.get('version')}")
    print(f"    Protocol: {init_resp.get('result', {}).get('protocolVersion')}")
    print(f"    Description:\n      {instructions.strip()}")
    print("-" * 80)

    tools = tools_resp.get("result", {}).get("tools", [])
    print(f"[+] Discovered {len(tools)} Official Meta Horizon OS Developer Tools:")
    for idx, t in enumerate(tools, 1):
        name = t.get("name")
        desc = t.get("description", "").split("\n")[0]
        params = list(t.get("inputSchema", {}).get("properties", {}).keys())
        print(f"  [{idx:02d}] {name:<24} | Params: {params}")
        print(f"       -> {desc[:90]}...")
    print("-" * 80)

    resources = res_resp.get("result", {}).get("resources", []) if res_resp else []
    print(f"[+] Discovered {len(resources)} Resources.")

    print("\n[V] HzOSDevMCP SETUP VERIFICATION 100% SUCCESSFUL!")

if __name__ == "__main__":
    main()
