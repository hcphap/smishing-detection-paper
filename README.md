\# Smishing and Suspicious SMS Detection



Hệ thống phát hiện tin nhắn lừa đảo (Smishing) và tin nhắn rác (SMS Spam) áp dụng các kiến trúc Học sâu nhẹ (Lightweight Deep Learning) và Học máy truyền thống. 



Thực hiện bởi: \*\*Huỳnh Công Pháp\*\*



\---



\## 📌 Cấu trúc dự án



```text

.

├── data/                    # Hướng dẫn tải và lưu trữ tập dữ liệu mẫu

├── models/                  # Lưu trữ checkpoint trọng số mô hình

├── src/                     # Mã nguồn chính của dự án

│   ├── preprocess.py        # Đọc dữ liệu, làm sạch văn bản và mã hóa

│   ├── model.py             # Định nghĩa kiến trúc mô hình (CNN-1D, BiLSTM, MaskedBiGRU)

│   └── evaluate.py          # Đo đạc chỉ số (F1, AUC, AP), độ trễ CPU và dung lượng

├── notebooks/               # Google Colab notebook chạy thử nghiệm toàn bộ

│   └── main\_demo.ipynb

├── paper/                   # Báo cáo/Bài báo khoa học liên quan

├── .gitignore

├── LICENSE                  # Giấy phép mở (MIT)

├── requirements.txt         # Danh sách thư viện cần thiết

└── README.md                # Tài liệu hướng dẫn dự án

