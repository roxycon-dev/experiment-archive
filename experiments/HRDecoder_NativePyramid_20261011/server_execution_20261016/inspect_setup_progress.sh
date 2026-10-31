BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
date -u
stat -c '%n %s' "$BASE/packages/data.tar" "$BASE/assets/hrnetv2_w48-d2186c55.pth"
ps -eo pid,etime,pcpu,args | grep -E 'pip install|remote_ops|install_env|python -m pip' | grep -v grep || true
find "$BASE/tmp" -name '*.whl' -printf '%s %p\n' 2>/dev/null | sort -nr | head -8
test ! -e "$BASE/receipts/environment_ready" || cat "$BASE/receipts/pip_freeze.txt" | grep -E 'torch|timm|numpy|opencv|Pillow'
df -h "$BASE"
