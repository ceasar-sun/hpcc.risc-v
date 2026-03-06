#!/usr/bin/env bash

# 檢查 slurm.conf 是否有設定 HealthCheckInterval 和 HealthCheckProgram
# How to use : 
# Add the 2 lines into slurm.conf
# >> HealthCheckInterval=30
# >> HealthCheckProgram=/usr/sbin/slurm_health_check.sh
#
#

# 設定要檢查的節點名稱（預設為本機 hostname）
NODE_NAME=$(hostname)

# default
_SLURM_HEAD_NODE=$(scontrol show config | grep -E "SlurmctldHost|ControlMachine" | awk -F'= ' '{print $2}')
#_SLURM_HEAD_NODE="slurm-ctl"
_ERROR_SIGN=0
_VERBOSE=y


# 1. 抓取 Slurm 偵測到的實際記憶體 (RealMemory)
REAL_MEM=$(scontrol show node "$NODE_NAME" | grep -oP 'RealMemory=\K\d+')

# 2. 抓取 slurm.conf 中定義的配置記憶體 (CfgTRES 中的 mem)
# 這裡使用 grep 抓取 CfgTRES 這一行，再用 sed 提取 mem= 後面的數字
CONF_MEM=$(scontrol show node "$NODE_NAME" | grep "CfgTRES" | sed -E 's/.*mem=([0-9]+)M.*/\1/')

# 檢查變數是否抓取成功
if [ -z "$REAL_MEM" ] || [ -z "$CONF_MEM" ]; then
    echo "錯誤: 無法從 scontrol 取得節點 $NODE_NAME 的記憶體資訊。"
    _ERROR_SIGN=1
fi


if [ "$_VERBOSE" = "y" ]; then
    echo "--- 記憶體檢查報告 ($NODE_NAME) ---"
    echo "Slurm 偵測實際值 (RealMemory): ${REAL_MEM} MB"
    echo "slurm.conf 定義值 (CfgTRES)  : ${CONF_MEM} MB"
fi

# 3. 邏輯比對
# 如果 實際偵測值 小於 設定要求值，Slurm 就會讓節點保持 DOWN
if [ "$REAL_MEM" -lt "$CONF_MEM" ]; then
    echo "[警告] 記憶體不足！實際值 ${REAL_MEM}MB 低於設定值 ${CONF_MEM}MB。"
    echo "這會導致 ReturnToService=1 失效。"
    _ERROR_SIGN=2
else
    [ "$_VERBOSE" = "y" ] && echo "[成功] 記憶體符合設定需求。"
fi

if [ "$HOSTNAME" != "$_SLURM_HEAD_NODE" ] ; then 
    # 範例：檢查 NFS 掛載是否存在
    if ! mountpoint -q /shared; then
        echo "NFS /shared mount missing"
        _ERROR_SIGN=2
    elif [ "$_VERBOSE" = "y" ] ; then 
        echo "[成功] NFS 掛載成功。"
    fi
else 
    echo "[ SKIP ] Skip NFS check at ${_SLURM_HEAD_NODE}"
fi

# 如果 實際偵測值 小於 設定要求值，Slurm 就會讓節點保持 DOWN
if [ "$_ERROR_SIGN" -eq 0 ]; then
    echo "[成功] 記憶體符合設定需求。"
else
    echo "[警告] 請確認上述錯誤是否排除！？"
    echo "導致 ReturnToService=1 失效。"
    exit $_ERROR_SIGN 
fi

exit 0
