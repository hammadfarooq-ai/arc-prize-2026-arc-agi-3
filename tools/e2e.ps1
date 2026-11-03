# End-to-end test mirroring the Kaggle rerun cell: local competition gateway + framework main.py
param([int]$BudgetS = 180)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $root "runs\e2e_agents"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
Copy-Item -Recurse (Join-Path $root "data\ARC-AGI-3-Agents") $work
Copy-Item (Join-Path $root "agent\my_agent.py") (Join-Path $work "agents\templates\my_agent.py")
Copy-Item (Join-Path $root "agent\explorer.py") (Join-Path $work "agents\templates\explorer.py")
@"
from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent
load_dotenv()
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}
"@ | Set-Content -Encoding utf8 (Join-Path $work "agents\__init__.py")
@"
SCHEME=http
HOST=127.0.0.1
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://127.0.0.1:8001/
OPERATION_MODE=online
RECORDINGS_DIR=$($work -replace '\\','/')/recordings
"@ | Set-Content -Encoding utf8 (Join-Path $work ".env")

$env:PYTHONPATH = Join-Path $root ".pylibs"
$env:ONLY_RESET_LEVELS = "true"
$server = Start-Process python -ArgumentList "`"$root\tools\serve.py`"" -PassThru -WindowStyle Hidden -RedirectStandardOutput "$root\runs\serve.out" -RedirectStandardError "$root\runs\serve.err"
Start-Sleep -Seconds 4
try {
    $env:AGENT_BUDGET_S = "$BudgetS"
    $env:MPLBACKEND = "agg"
    Push-Location $work
    python main.py --agent myagent 2>&1 | Select-String -Pattern "WARNING|ERROR|Traceback|Error|score|SCORECARD" | Select-Object -Last 40
    Pop-Location
} finally {
    Start-Sleep -Seconds 2
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}
Get-Content "$root\runs\serve.out" -Tail 5
