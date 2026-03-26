# Local Test Setup Guide (Windows)

## 🎯 Hızlı Başlangıç

### 1. Gerekli Araçlar

#### Python & Poetry
```powershell
# Python 3.10+ kontrol et
python --version

# Poetry yoksa yükle
pip install poetry
```

#### Docker (Opsiyonel - Docker Compose için)
- Docker Desktop for Windows: https://www.docker.com/products/docker-desktop

### 2. API Key'leri Al

#### Pinecone API Key
1. https://www.pinecone.io/ adresine git
2. Sign up / Login yap
3. Dashboard'da **API Keys** bölümüne git
4. API key'ini kopyala

#### Google API Key (Gemini API için)
1. https://aistudio.google.com/app/apikey adresine git
2. Google hesabınla login ol
3. **Create API Key** butonuna tıkla
4. API key'ini kopyala

**VEYA** Vertex AI kullanmak istersen:
- GCP hesabı aç (https://cloud.google.com/)
- $300 free credit alırsın
- Vertex AI API'yi enable et

### 3. .env Dosyası Oluştur

Proje root dizininde `.env` dosyası oluştur:

```powershell
# Windows PowerShell'de
New-Item -Path .env -ItemType File
```

Sonra içine şunları yaz:

```env
# Pinecone
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=video-games

# Google Gemini API (Development için)
USE_VERTEX_AI=false
GOOGLE_API_KEY=your-google-api-key-here

# VEYA Vertex AI (Production için)
# USE_VERTEX_AI=true
# GOOGLE_PROJECT_ID=your-gcp-project-id
# GOOGLE_LOCATION=us-central1

# Tracing (Local Phoenix için)
ENABLE_TRACING=true
PHOENIX_ENDPOINT=http://localhost:6006

# API Settings
LOG_LEVEL=INFO
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60
```

### 4. Dependencies Yükle

```powershell
# Poetry ile dependencies yükle
poetry install

# Veya pip ile (eğer poetry kullanmıyorsan)
pip install -r requirements.txt
```

### 5. Local Test

#### Seçenek A: Poetry ile Direkt Çalıştır

```powershell
# API'yi başlat
poetry run uvicorn src.api.main:app --reload --port 8000

# Başka bir terminal'de Phoenix'i başlat (opsiyonel)
poetry run phoenix serve
```

#### Seçenek B: Docker Compose ile

```powershell
# Docker Desktop'ı başlat
# Sonra:
docker-compose up

# API: http://localhost:8000
# Phoenix: http://localhost:6006
```

### 6. Test Et

#### PowerShell'de:

```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:8000/health

# Query test
$body = @{
    query = "What are the best RPG games on Nintendo Switch?"
    top_k = 5
    platform = "Switch"
    use_agent = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/api/v1/query `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

#### Browser'da:
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Phoenix UI: http://localhost:6006 (eğer çalışıyorsa)

## 🔑 Credential'lar Nereye Gidiyor?

### .env Dosyası (Local Development)
- **Konum**: Proje root dizini (`C:\Users\u26c96\Desktop\ADAS\10_Personal\07_AI\05_Video_Game_RAG\.env`)
- **Format**: `KEY=value` (her satır bir key-value pair)
- **Güvenlik**: `.env` dosyası `.gitignore`'da olmalı (git'e commit edilmemeli)

### GCP Secret Manager (Production)
- **Konum**: Google Cloud Console → Secret Manager
- **Nasıl Oluşturulur**: `deploy.sh` script'i otomatik oluşturur
- **Nasıl Kullanılır**: Cloud Run deployment'da `--set-secrets` ile

### Environment Variables (Cloud Run)
- **Konum**: Cloud Run service configuration
- **Nasıl Set Edilir**: `gcloud run deploy` komutunda `--set-env-vars` ile

## 🐛 Troubleshooting

### "Module not found" hatası
```powershell
# Poetry environment'ı aktif et
poetry shell

# Veya direkt poetry run ile çalıştır
poetry run uvicorn src.api.main:app --reload
```

### "API key not found" hatası
- `.env` dosyasının proje root'unda olduğundan emin ol
- API key'lerin doğru olduğunu kontrol et
- `.env` dosyasında boşluk olmamasına dikkat et: `KEY=value` (boşluk yok)

### Pinecone connection hatası
- Pinecone index'in oluşturulduğundan emin ol
- Index name'in `.env`'deki ile aynı olduğunu kontrol et
- API key'in doğru olduğunu doğrula

### Phoenix çalışmıyor
- Phoenix opsiyonel, tracing olmadan da API çalışır
- `ENABLE_TRACING=false` yaparak devre dışı bırakabilirsin

## 📝 Önemli Notlar

1. **.env dosyası git'e commit edilmemeli**
   - `.gitignore`'da olmalı
   - `.env.example` template olarak kullanılabilir

2. **Phoenix hesabı açmana gerek yok**
   - Phoenix self-hosted (ücretsiz)
   - Local'de `phoenix serve` ile çalıştırılır
   - Cloud'da Cloud Run'da deploy edilir

3. **GCP hesabı sadece production için gerekli**
   - Local test için Gemini API yeterli
   - Vertex AI için GCP hesabı + $300 credit gerekli

4. **Windows'ta path'ler**
   - PowerShell'de `\` yerine `/` kullanabilirsin
   - Veya `\` escape et: `\\`

## 🚀 Sonraki Adımlar

1. ✅ .env dosyası oluştur
2. ✅ API key'leri ekle
3. ✅ `poetry install` çalıştır
4. ✅ `poetry run uvicorn src.api.main:app --reload` ile başlat
5. ✅ http://localhost:8000/docs adresine git
6. ✅ Test query gönder

Başarılar! 🎉

