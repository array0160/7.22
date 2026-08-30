# BookOCR Web v7.2 Experimental — Ink Guard + Stable Vertical Reading Order

v7.2 修兩個使用者直接指出的問題：

1. 相鄰欄被錯分成 `9` / `9.2`，導致閱讀順序不合理。
2. 空白／背面透字被 Detector 框到後，Recognizer 在根本沒有正面文字的區域 hallucinate 中文字。

另外依這批老書實測：
`頁面安全留白` 預設由 3.0% 改為 0.5%。

---

## 1. 為什麼 9 / 9.2 會錯

舊 grouping 主要依：

- x center 接近
- x 區間 overlap

判斷兩個 Detector boxes 是否同一欄。

問題是傳統直書的兩個**相鄰欄**本來 x 就很近。
頁面彎曲後甚至會有 x overlap。

因此左右兩個欄可能被誤當成：
`同一欄上段 + 下段`

接著系統依 y 排序，
就可能產生使用者看到的：
`9.2 → 9`

### v7.2 新規則

真正同一欄被切兩段：

- 應該上下接續
- y overlap 很少

如果兩個 boxes 在 y 方向重疊 >= 約 28%：

**絕對不合併。**

直接視為左右兩欄。

只有：
- x 接近
- y 幾乎不重疊
- 上下 gap 合理

才允許合成同一 column。

最終：
- Columns：x 右 → 左
- 同欄 fragments：box top y 上 → 下

畫面編號與全文輸出使用完全同一套排序。

---

## 2. Ink Guard：空白處不應該送 OCR

Recognizer 本身的工作是：

`給我一張文字 crop，我一定嘗試輸出文字`

所以如果 Detector 在：
- 紙張陰影
- 背面透字
- 淡污漬
- 空白
上產生 false-positive box，

Recognizer 很可能仍會硬猜出：
`酬明耐国束美`
之類不存在的字。

只看 OCR confidence 不夠，
因為 neural recognizer 對 hallucination 也可能有 confidence。

### v7.2 Recognition 前直接看像素

每一個 Detector quad 先量：

- 紙張背景亮度
- 10% / 92% grayscale percentile contrast
- 強黑筆畫比例
- 中等墨跡比例
- weighted darkness
- 12 個垂直區段裡，有幾段真的存在強墨跡

如果缺乏真正前景：

`Detector box`
→ `Ink Guard Reject`
→ **不送 Recognition**

所以不是 OCR 後才刪錯字，
而是根本不讓模型在空白上猜。

### 第二遍 Detector 更嚴格

第二遍本來就是：
`1.2× + 敏感 threshold`

它最容易抓到透字與紙紋。

因此 v7.2：

- Primary first-pass box：較寬鬆
- Recovery second-pass box：要求更明顯墨跡

這樣保留補漏能力，
但降低 17.2 / 17.3 類 false positive。

---

## 3. 安全留白預設 0.5%

使用者實測：

`0.5%`
比
`3%`

對這張頁面的對欄效果更好。

這是合理的：

過多 padding 會使真正頁面內容在整張 UVDoc input 中相對縮小，
後續 detector resize / DB threshold 更容易跨過臨界值。

所以 v7.2 預設：

`頁面安全留白 = 0.5%`

仍可手動調整。

---

## Ink Guard 選項

### 關閉
所有 Detector boxes 都 OCR。

### 平衡（預設）
- Primary 輕度 filter
- Recovery 比較嚴格

### 嚴格
適合背面透字很重的頁面。
但非常淡的正面印刷可能被漏掉，所以不是預設。

---

## 建議測這張圖

設定：

- 安全留白 0.5%
- Hybrid = 平衡
- Ink Guard = 平衡

看：

1. `同書又述南路…` 的右欄先出
2. `南路番童習漢書者…` 下一欄才出
3. 不應再把兩個大量 y-overlap 的欄標成 `9 / 9.2`
4. 17 下方兩個空白 false boxes 是否消失
5. 全文最後不應再冒出不存在的 `酬明耐国束美`

舊 V3 完整保留。
