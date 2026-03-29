# Dataset layout · 数据目录约定

## 中文

将本仓库克隆后，在**项目根目录**自行创建 `data/`（该目录默认被 `.gitignore` 忽略，不上传大图）。

推荐结构：

```text
data/
├── train/
│   ├── images/          # 训练图片，如 .jpg
│   └── annotations/     # Pascal VOC，与图片同名的 .xml
└── test/
    └── images/          # 仅推理用的测试图（可无标注）
```

规则：

- 每张图对应一个 XML：`img_0001.jpg` ↔ `img_0001.xml`
- XML 中为 `<object>` + `<name>` + `<bndbox>`（xmin, ymin, xmax, ymax）

预处理脚本默认读取：

- 图片：`data/train/images/`
- 标注：`data/train/annotations/`

输出：`yolo_dataset/`（同样默认不入库）

---

## English

After cloning, create `data/` at the **project root** locally (`data/` is gitignored).

Recommended layout:

```text
data/
├── train/
│   ├── images/          # Training images (e.g. .jpg)
│   └── annotations/     # Pascal VOC .xml, same basename as image
└── test/
    └── images/          # Optional test images (no labels required)
```

Rules:

- One XML per image: `img_0001.jpg` ↔ `img_0001.xml`
- VOC `<object>`, `<name>`, `<bndbox>` (xmin, ymin, xmax, ymax)

The prepare script expects:

- Images: `data/train/images/`
- Annotations: `data/train/annotations/`

Output: `yolo_dataset/` (also gitignored by default)
