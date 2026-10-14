import pdfplumber
import pandas as pd
import re

pdf_path = "Lista Nomes Próprios.pdf"   

female_names = []
male_names = []

def clean_name(name):
    """Clean extracted text."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        # Extract table data from page
        tables = page.extract_tables()

        for table in tables:
            for row in table:
                if not row or len(row) < 2:
                    continue

                gender = str(row[0]).strip().lower()
                name = clean_name(str(row[1]))

                # Skip headers
                if "género" in gender or "genero" in gender:
                    continue
                if name.lower() == "nome":
                    continue

                # Detect feminine
                if "femin" in gender:
                    female_names.append(name)

                # Detect masculine
                elif "mascul" in gender:
                    male_names.append(name)

pd.DataFrame({"name": female_names}).to_csv(
    "nomes_femininos.csv", index=False, encoding="utf-8-sig"
)

pd.DataFrame({"name": male_names}).to_csv(
    "nomes_masculinos.csv", index=False, encoding="utf-8-sig"
)

print("Done!")
print(f"Feminine names: {len(female_names)}")
print(f"Masculine names: {len(male_names)}")