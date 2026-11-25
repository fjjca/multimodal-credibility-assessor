# 演示图片来源

以下图片用于本项目首页"演示"按钮与 `data/multimodal_photo_demo.jsonl` 评测用例，均为**真实网络照片**，
经等比缩放（最长边 ≤ 1024px、JPEG q88）后存于本目录。

| 文件 | 内容 | 原始来源 |
|---|---|---|
| `paris_eiffel.jpg` | 巴黎埃菲尔铁塔 | http://b0.bdstatic.com/a6635005d706780b4b15385fb31ed72f.jpg |
| `beijing_marathon.jpg` | 2024 北京半程马拉松起跑现场（拱门有赛事文字） | http://q6.itc.cn/q_70/images03/20240414/c1c96323690d4ddf9cc3ae68c4e3f557.jpeg |
| `supermarket_receipt.jpg` | 超市小票（可见"合计金额 150.00"） | http://b0.bdstatic.com/b15f93965018b0c4723de74f422ebd52.jpg |

## 授权说明（重要）

这些图片来自国内新闻/图片站点的公开 CDN，**并非明确采用 CC0 / CC-BY 等开放许可**，
仅用于本地演示与效果验证（含评测数据）。

- 若仓库要公开分发，建议二选一：
  1. 把本目录加入 `.gitignore`（当前默认不提交，见下），只在本机保留；
  2. 替换为明确开放许可的图片（如 Wikimedia Commons 上 CC0/CC-BY 素材），并在 `SOURCES.md` 标注作者与许可。
- 替换时只需保持文件名不变，`static/index.html` 的 `/demo_photos/<文件名>` 引用与
  `data/multimodal_photo_demo.jsonl` 的 `image` 字段无需改动。

## 生成方式

原始图下载后用 Pillow 统一缩放与压缩：

```python
from PIL import Image
im = Image.open(src).convert("RGB")
im.thumbnail((1024, 1024), Image.LANCZOS)
im.save(dst, "JPEG", quality=88)
```
