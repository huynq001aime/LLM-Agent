FROM python:3.13.2-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Nếu đưa lên github action rồi deploy lên server thì cmt dòng COPY .env .
# COPY .env .

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
