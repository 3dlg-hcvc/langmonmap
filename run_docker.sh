source .env

# Run appropriate docker-compose command based on DOWNLOAD_SET
if [ "$HM3D" = "LOCAL" ]; then
    echo "Using local dataset..."
    docker compose -f docker-compose.yaml -f docker-compose.local.yaml up -d
else
    echo "Using downloaded dataset..."
    docker compose up -d
fi
