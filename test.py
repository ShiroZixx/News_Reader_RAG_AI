

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

text = """
đào thải ra ngoài qua đường bài tiết.Khi dịch mật bị đào thải, gan buộc phải sản xuất lượng mật mới để bù đắp. Để có nguyên liệu sản xuất, gan sẽ tăng cường tìm kiếm cholesterol dư thừa trong máu bằng cách tăng các thụ thể LDL. Kết quả là nồng độ LDL trong máu sẽ giảm xuống đáng kể.
"""
tokens = encoding.encode(text)

print(len(tokens))


