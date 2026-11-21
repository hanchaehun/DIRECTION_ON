# 행사장 네트워크 준비 (관리자 PowerShell에서 실행). 자동으로 돌리지 말고 필요한 줄만 직접 실행한다.
#
# 1) 현재 상태 보기: 어떤 인터페이스가 기본 경로인지 (Wi-Fi와 USB 테더링을 함께 연결했을 때 중요)
Get-NetIPConfiguration | Format-Table InterfaceAlias, IPv4Address, IPv4DefaultGateway -AutoSize
Get-NetRoute -DestinationPrefix 0.0.0.0/0 | Sort-Object RouteMetric | Format-Table InterfaceAlias, NextHop, RouteMetric -AutoSize

# 2) 방화벽: 상담원 서버(9000)와 모의 함수 서버(8100)를 개인·공용 프로필 모두 허용
#    (행사장 Wi-Fi는 보통 '공용' 네트워크로 잡힌다)
# New-NetFirewallRule -DisplayName "DIRECTON agent 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow -Profile Private,Public
# New-NetFirewallRule -DisplayName "DIRECTON mock 8100"  -Direction Inbound -Protocol TCP -LocalPort 8100 -Action Allow -Profile Private,Public

# 3) 운영진 중계(tools_url) 주소가 행사장 내부 주소이고, 다른 서브넷이라 기본 경로(테더링)로 나가 버릴 때:
#    중계 IP만 Wi-Fi 게이트웨이로 보내는 경로를 추가한다. <중계IP>, <Wi-Fi게이트웨이>를 1)의 결과로 채운다.
# route add <중계IP> mask 255.255.255.255 <Wi-Fi게이트웨이>
#    확인:  Test-NetConnection <중계IP> -Port <중계포트>
#    삭제:  route delete <중계IP>

# 4) 다른 노트북에서 접속 시험 (팀원 노트북에서)
# Test-NetConnection <내IP> -Port 9000
# Test-NetConnection <내IP> -Port 8100
