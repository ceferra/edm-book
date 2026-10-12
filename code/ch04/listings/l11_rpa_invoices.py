import csv
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation

EMAILS = [  # the inbox the robot reads every morning
    "From: Ferreteria Sol\nSubject: Invoice INV-2041\nDate: 12/03/2026\n"
    "Total: 1.234,50 EUR",
    "From: Papeleria Mar\nSubject: Factura F-778\nFecha: 2026-03-15\nImporte: 89,90 EUR",
    "From: Tech Supplies Ltd\nSubject: Invoice no. 5531\nDate: 15 March 2026\n"
    "Amount due: EUR 12,480.00",
    "From: Limpiezas Turia\nSubject: Monthly service\nDate: 18/03/2026\n"
    "Total: 640,00 EUR",
    "From: Imprenta Rapida\nSubject: Factura F-2026-31\nFecha: 20/03/2026\n"
    "Total: 312,75 EUR",
    "From: Acme, SL\nSubject: Invoice INV-77\nDate: 31/02/2026\nTotal: 45,00 EUR"]
LIMIT = Decimal(10_000)                            # larger amounts need a manager

def parse_date(text):                              # None if impossible or unknown
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %B %Y"):   # month names: English locale
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            pass

def parse_amount(line):                            # "1.234,50 EUR" -> (1234.50, "EUR")
    number = re.search(r"(-?)(\d[\d.,]*\d)", line)
    currency = re.search(r"\b(EUR|USD|GBP)\b", line)
    if number is None or currency is None:
        return None, None
    digits = number.group(2)
    if re.search(r",\d{2}$", digits):              # decimal comma
        digits = digits.replace(".", "").replace(",", ".")
    try:
        return Decimal(number.group(1) + digits.replace(",", "")), currency.group(1)
    except InvalidOperation:
        return None, None

out = csv.writer(sys.stdout, lineterminator="\n")  # quotes fields when needed
out.writerow(["invoice", "date", "supplier", "amount", "status"])
for email in EMAILS:                               # every message is handled on its own
    supplier = re.search(r"From: (.+)", email)
    number = re.search(r"(?:Invoice|Factura)(?: no\.)? ([A-Z]+-[\d-]+|\d+)", email)
    date = re.search(r"(?:Date|Fecha): (.+)", email)
    date = parse_date(date.group(1)) if date else None
    amount, currency = parse_amount(email.splitlines()[-1])
    if number is None or supplier is None:
        status = "to human: no invoice number"
    elif date is None:
        status = "to human: missing or invalid date"
    elif amount is None or currency != "EUR" or amount <= 0:
        status = "to human: amount or currency not supported"
    elif amount > LIMIT:
        status = "to human: needs approval"
    else:
        status = "ready for booking"
    out.writerow([number.group(1) if number else "?", date or "?",
                  supplier.group(1) if supplier else "?",
                  f"{amount:.2f}" if amount is not None else "?", status])
