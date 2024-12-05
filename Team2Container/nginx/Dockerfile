FROM nginx:latest

ENV NGINX_HOME=/usr/share/nginx

WORKDIR $NGINX_HOME

COPY nginx.conf /etc/nginx/nginx.conf

# 애플리케이션 파일 복사 및 권한 설정
COPY . $NGINX_HOME

# RUN chmod -R 755 $NGINX_HOME
# RUN chown -R www-data:www-data /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]