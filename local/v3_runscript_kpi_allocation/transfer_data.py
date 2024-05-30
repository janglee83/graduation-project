import pandas as pd

# Đọc tệp CSV
# Thay thế bằng đường dẫn tệp CSV của bạn
csv_file_path = 'final_solution.csv'
df = pd.read_csv(csv_file_path)

# Chuyển đổi dấu thập phân từ dấu chấm sang dấu phẩy
df = df.applymap(lambda x: str(x).replace('.', ',')
                 if isinstance(x, (float, int)) else x)

# Lưu thành tệp Excel
# Thay thế bằng đường dẫn tệp Excel đầu ra của bạn
excel_file_path = 'final_solution.xlsx'
df.to_excel(excel_file_path, index=False)
