from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="KariyerAI",
    description="Yapay zekâ destekli CV ve iş/staj başvuru asistanı",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
 GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

def ask_ollama(prompt):

    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY bulunamadı.")

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return json.loads(
        data["choices"][0]["message"]["content"]
    )

def extract_pdf_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def calculate_ats_score(cv_text):
    text = cv_text.lower()

    checks = {
        "iletisim": any(
            x in text
            for x in ["@", "linkedin", "github"]
        ),

        "egitim": any(
            x in text
            for x in [
                "education",
                "eğitim",
                "üniversite",
                "university"
            ]
        ),

        "beceriler": any(
            x in text
            for x in [
                "skills",
                "beceriler",
                "technical skills"
            ]
        ),

        "projeler": any(
            x in text
            for x in [
                "projects",
                "projeler",
                "project"
            ]
        ),

        "deneyim": any(
            x in text
            for x in [
                "experience",
                "deneyim",
                "internship",
                "staj"
            ]
        )
    }

    passed = sum(checks.values())

    score = round(
        passed / len(checks) * 100
    )

    return score, checks


@app.get("/")
def home():
    return {
        "uygulama": "KariyerAI",
        "surum": "2.1",
        "durum": "çalışıyor"
    }


@app.post("/analyze-cv")
async def analyze_cv(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Lütfen PDF formatında bir CV yükleyin."
        )

    if len(job_description.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="İş ilanı açıklaması çok kısa."
        )

    pdf_bytes = await file.read()

    cv_text = extract_pdf_text(pdf_bytes)

    if not cv_text:
        raise HTTPException(
            status_code=400,
            detail="CV içerisindeki metin okunamadı."
        )

    # ------------------------------------------------
    # 1. İLANDAN BECERİLERİ ÇIKAR
    # ------------------------------------------------

    skill_prompt = f"""
Bir işe alım uzmanısın.

Aşağıdaki iş veya staj ilanındaki
teknik becerileri çıkar.

İLAN:

{job_description}

İki kategori oluştur:

1. zorunlu
2. tercih

Takım çalışması, iletişim,
öğrenmeye açıklık gibi kişisel
özellikleri dahil etme.

Sadece JSON döndür.

{{
  "zorunlu": ["Python", "Linux"],
  "tercih": ["AWS", "Docker"]
}}
"""

    skills = ask_ollama(skill_prompt)

    required = skills.get("zorunlu", [])
    preferred = skills.get("tercih", [])

    # ------------------------------------------------
    # 2. CV'DE BECERİLERİ KONTROL ET
    # ------------------------------------------------

    match_prompt = f"""
Bir CV doğrulama sistemi olarak çalışıyorsun.

CV:

{cv_text}

Zorunlu beceriler:
{required}

Tercih edilen beceriler:
{preferred}

Her beceri için CV'de açık kanıt olup
olmadığını belirle.

Kurallar:

- Tahmin yapma.
- CV'de olmayan bir beceriyi varmış gibi gösterme.
- Benzer teknolojileri aynı şey kabul etme.
- Projede gerçekten kullanılmışsa geçerli say.
- true veya false döndür.

Sadece JSON döndür.

{{
  "zorunlu": {{
    "Python": true,
    "Linux": false
  }},
  "tercih": {{
    "AWS": false
  }}
}}
"""

    matches = ask_ollama(match_prompt)

    required_matches = matches.get(
        "zorunlu",
        {}
    )

    preferred_matches = matches.get(
        "tercih",
        {}
    )

    # ------------------------------------------------
    # 3. TEKNİK PUAN
    # ------------------------------------------------

    required_total = len(required_matches)
    preferred_total = len(preferred_matches)

    required_found = sum(
        value is True
        for value in required_matches.values()
    )

    preferred_found = sum(
        value is True
        for value in preferred_matches.values()
    )

    required_ratio = (
        required_found / required_total
        if required_total
        else 1
    )

    preferred_ratio = (
        preferred_found / preferred_total
        if preferred_total
        else 1
    )

    technical_score = round(
        required_ratio * 75
        +
        preferred_ratio * 25
    )

    # ------------------------------------------------
    # 4. ATS PUANI
    # ------------------------------------------------

    ats_score, ats_checks = calculate_ats_score(
        cv_text
    )

    # ------------------------------------------------
    # 5. GENEL PUAN
    # ------------------------------------------------

    overall_score = round(
        technical_score * 0.85
        +
        ats_score * 0.15
    )

    # ------------------------------------------------
    # 6. EŞLEŞEN / EKSİK BECERİLER
    # ------------------------------------------------

    matched_skills = []
    missing_skills = []

    for skill, result in required_matches.items():
        if result:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    for skill, result in preferred_matches.items():
        if result:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # ------------------------------------------------
    # 7. EKSİK BECERİLER İÇİN AÇIKLAMA
    # ------------------------------------------------

    missing_reason_prompt = f"""
Bir kariyer danışmanısın.

CV:

{cv_text}

İş ilanı:

{job_description}

Eksik beceriler:

{missing_skills}

Her eksik beceri için çok kısa şekilde
neden eksik kabul edildiğini açıkla.

Sadece CV'deki gerçek bilgilere dayan.

Sadece JSON döndür.

Format:

{{
  "aciklamalar": [
    {{
      "beceri": "AWS",
      "neden": "CV'de AWS kullanımına dair açık bir proje veya deneyim bulunamadı."
    }}
  ]
}}
"""

    missing_reason_data = ask_ollama(
        missing_reason_prompt
    )

    missing_reasons = missing_reason_data.get(
        "aciklamalar",
        []
    )

    # ------------------------------------------------
    # 8. KARİYER İÇERİKLERİ
    # ------------------------------------------------

    career_prompt = f"""
Sen profesyonel bir kariyer danışmanısın.

ADAY CV:

{cv_text}

İŞ İLANI:

{job_description}

GENEL UYUM:
%{overall_score}

TEKNİK UYUM:
%{technical_score}

ATS:
%{ats_score}

EŞLEŞEN BECERİLER:
{matched_skills}

EKSİK BECERİLER:
{missing_skills}

ÖNEMLİ KURALLAR:

- CV'de olmayan bir deneyimi uydurma.
- Adayın sahip olmadığı beceriyi
  CV'ye yazmasını önerme.
- Sadece gerçek bilgiler üzerinden öneri üret.
- CV önerilerini sadece düz metin olarak yaz.
- Tüm içerik Türkçe olsun.

Sadece JSON döndür.

Format:

{{
  "cv_onerileri": [
    "Öneri 1",
    "Öneri 2",
    "Öneri 3"
  ],

  "basvuru_tavsiyesi": "Başvurmalı",

  "degerlendirme": "Kısa değerlendirme",

  "basvuru_maili": "Kısa ve profesyonel başvuru maili",

  "cover_letter": "İlana özel kısa cover letter",

  "linkedin_mesaji": "Recruiter'a gönderilebilecek kısa LinkedIn mesajı",

  "mulakat_sorulari": [
    "Soru 1",
    "Soru 2",
    "Soru 3",
    "Soru 4",
    "Soru 5"
  ]
}}
"""

    career = ask_ollama(
        career_prompt
    )

    # ------------------------------------------------
    # 9. SONUÇ
    # ------------------------------------------------

    return {
        "dosya": file.filename,

        "genel_puan": overall_score,
        "teknik_puan": technical_score,
        "ats_puani": ats_score,

        "ats_kontrolleri": ats_checks,

        "zorunlu_beceriler": required,
        "tercih_edilen_beceriler": preferred,

        "eslesen_beceriler": matched_skills,
        "eksik_beceriler": missing_skills,

        "eksik_beceri_aciklamalari": missing_reasons,

        "karsilanan_zorunlu": required_found,
        "toplam_zorunlu": required_total,

        "karsilanan_tercih": preferred_found,
        "toplam_tercih": preferred_total,

        "cv_onerileri": career.get(
            "cv_onerileri",
            []
        ),

        "basvuru_tavsiyesi": career.get(
            "basvuru_tavsiyesi",
            ""
        ),

        "degerlendirme": career.get(
            "degerlendirme",
            ""
        ),

        "basvuru_maili": career.get(
            "basvuru_maili",
            ""
        ),

        "cover_letter": career.get(
            "cover_letter",
            ""
        ),

        "linkedin_mesaji": career.get(
            "linkedin_mesaji",
            ""
        ),

        "mulakat_sorulari": career.get(
            "mulakat_sorulari",
            []
        )
    }