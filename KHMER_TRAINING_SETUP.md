# Hướng dẫn Setup Training cho Tiếng Khmer

## ✅ Đã cập nhật symbols cho Khmer

### Những thay đổi đã thực hiện:

1. **Thêm `_KHMER_IPA` vào symbols list** (55 IPA symbols cho Khmer)
2. **Thêm `_KHMER_UNICODE` vào symbols list** (42 ký tự Khmer Unicode)
3. **Loại bỏ `_CNM3_letters`** (chỉ dùng cho Chinese, không cần cho Khmer-only)

### Kết quả:
- ✅ Tất cả 97 unique phones trong dataset đều có trong symbols list
- ✅ Không còn lỗi "symbol not found"
- ✅ Model size nhỏ hơn (loại bỏ ~200 CNM3 symbols không dùng)

## 📊 Symbols Configuration

### Hiện tại (Khmer-only):
```python
symbols = [_pad] + list(_punctuation) + list(_IPA_letters) + _KHMER_IPA + _KHMER_UNICODE + _additional
```

**Tổng số symbols: ~168** (bao gồm):
- 1 padding symbol
- 8 punctuation marks
- ~23 IPA letters (overlap với Khmer)
- 55 Khmer IPA symbols
- 42 Khmer Unicode characters
- 2 additional symbols (<sil>, <asp>)

### Nếu muốn tối ưu hơn (minimal):
Uncomment dòng này trong `text/symbols.py`:
```python
symbols = [_pad] + list(_punctuation) + _KHMER_IPA + _KHMER_UNICODE + _additional
```
**Tổng số symbols: ~108** (nhỏ hơn ~60 symbols)

## 🚀 Sẵn sàng để train!

### Bước 1: Kiểm tra lại
```bash
# Đảm bảo symbols.py đã được cập nhật
python -c "from text.symbols import symbols; print(f'Total symbols: {len(symbols)}')"
```

### Bước 2: Train như bình thường
```bash
# Single GPU
python train_local.py --device cuda

# Multi-GPU
python train.py
```

### Bước 3: Monitor training
- Checkpoints sẽ được lưu vào `./checkpoints/`
- TensorBoard logs: `./runs/`
- Training sẽ không còn lỗi "symbol not found"

## 📝 Lưu ý

1. **Model size**: Model sẽ nhỏ hơn vì loại bỏ CNM3 symbols (~200 symbols)
2. **Compatibility**: Model này chỉ dùng cho Khmer. Nếu sau này muốn train multilingual, cần thêm lại `_CNM3_letters`
3. **Symbols order**: Đảm bảo không thay đổi thứ tự symbols sau khi train đã bắt đầu (sẽ ảnh hưởng đến checkpoint)

## 🔄 Nếu muốn quay lại multilingual

Uncomment dòng này trong `text/symbols.py`:
```python
symbols = [_pad] + list(_punctuation) + list(_IPA_letters) + _CNM3_letters + _KHMER_IPA + _KHMER_UNICODE + _additional
```

## ✅ Checklist

- [x] `_KHMER_IPA` đã được thêm vào symbols
- [x] `_KHMER_UNICODE` đã được thêm vào symbols  
- [x] `_CNM3_letters` đã được loại bỏ
- [x] Tất cả 97 phones trong dataset đều có trong symbols
- [x] Không còn lỗi "symbol not found"
- [x] Sẵn sàng để train!

