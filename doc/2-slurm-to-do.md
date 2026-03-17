# Slurm 叢集管理與優化建議 (slurm-to-do.md)

## 原始需求 (Original Request)

分析 `/etc/slurm/slurm.conf` 與服務啟動，提供以下：
1. 檢查 `NodeName` 與 `PartitionName` 區段是否有可優化之處，如: `State=UNKNOWN` 寫法。
2. 如何優化，使其有新計算節點加入時，可自動化加入減少人為介入。
3. 提供建議，在需要條件下讓 job 無法送至 `slurmc02`, `slurmc03` 節點。

---

## 1. NodeName 與 PartitionName 區段優化 (關於 State=UNKNOWN)

在早期或預設的教學中，經常會看到設定 `State=UNKNOWN`。這代表當 `slurmctld` 啟動時不知道節點的硬體狀態，會等待各運算節點的 `slurmd` 啟動並回報真實的硬體規格後，才將節點納入可用資源。

### 優化建議：
* **明確定義硬體規格（推薦）：** 建議**明確寫出預期的硬體資源** (如 `CPUs=... RealMemory=... Sockets=...`)，並搭配 `ReturnToService=1`。如果節點回報的資源小於 `slurm.conf` 中設定的值，節點會被判定為 `DRAIN` 狀態，這是一種保護機制的最佳實踐。
* **使用節點範圍縮寫：** 如果您的節點命名是有規律的，強烈建議使用範圍表示法來精簡設定檔。
  ```ini
  # 優化寫法示例 (明確規格、簡潔)
  NodeName=slurmc[01-10] CPUs=8 RealMemory=16000 Sockets=1 CoresPerSocket=4 ThreadsPerCore=2 State=UNKNOWN
  ```
* **PartitionName 優化：** 同樣使用縮寫，並且建議加上預設時間限制 (`DefaultTime`) 或最大時間限制 (`MaxTime`)。
  ```ini
  PartitionName=normal Nodes=slurmc[01-10] Default=YES MaxTime=24:00:00 State=UP
  ```

---

## 2. 自動化新節點加入機制 (減少人為介入)

要達成自動化加入，有兩種主流方法：

### 方法 A：動態節點註冊 (Dynamic Node Registration) - 適用於 Slurm 22.05+
Slurm 最新的版本支援完全動態的節點加入，不需事前寫死所有節點名字。
1. 在 `slurm.conf` 啟用動態節點功能：
   ```ini
   SlurmctldParameters=enable_configless
   MaxNodeCount=512 
   ```
2. 在 `slurm.conf` 加入供動態節點使用的定義：
   ```ini
   NodeName=DEFAULT Feature=dynamic_node State=CLOUD
   ```
3. 當新的 VM 或實體機啟動 `slurmd -Z --conf "NodeName=... CPUs=... RealMemory=..."`，它會動態註冊並自動加入。

### 方法 B：預先註冊未來節點 (FUTURE State) - 適用於所有版本 (推薦)
1. 在 `slurm.conf` 預先宣告未來可能擴充的節點，並將狀態設為 `FUTURE`：
   ```ini
   NodeName=slurmc[01-10] CPUs=8 RealMemory=16000 State=UNKNOWN  # 現有節點
   NodeName=slurmc[11-50] CPUs=8 RealMemory=16000 State=FUTURE   # 未來節點
   PartitionName=normal Nodes=slurmc[01-50] Default=YES State=UP
   ```
2. 設定為 `FUTURE` 的節點不會報錯，也不會配發 Job。
3. **自動化加入邏輯：** 當新節點安裝好並啟動 `slurmd` 後，只要下達指令即可讓它上線（可寫成開機註冊腳本）：
   ```bash
   scontrol update NodeName=slurmc11 State=RESUME
   ```

---

## 3. 特定條件下禁止 Job 派送至特定節點 (slurmc02, slurmc03)

依據情境提供以下三種作法：

### 作法 A：臨時維護或硬體異常 (DRAIN)
如果您是因為維護狀況不讓 Job 進入：
```bash
# 設定節點為 DRAIN：已經在跑的 Job 會跑完，但不再接受新 Job。
scontrol update NodeName=slurmc[02-03] State=DRAIN Reason="Hardware maintenance"

# 恢復節點：
scontrol update NodeName=slurmc[02-03] State=RESUME
```

### 作法 B：保留給特定使用者或專案 (Reservation)
如果這兩台機器要保留給特定使用者使用，其他人的 Job 會自動避開：
```bash
# 建立預約，只有使用者 'alice' 能使用這兩台：
scontrol create reservation starttime=now duration=infinite user=alice nodes=slurmc[02-03] name=special_reserve

# 移除限制：
scontrol delete ReservationName=special_reserve
```

### 作法 C：基於 Job 屬性隔離 (Node Features)
1. 在 `slurm.conf` 中給這些節點上標籤 (`Feature`)：
   ```ini
   NodeName=slurmc[02-03] CPUs=8 Feature=special_hardware
   ```
2. 除非使用者在 Job Script 內宣告 `#SBATCH --constraint=special_hardware`，否則一般 Job 不會被派發過去。
