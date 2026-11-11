"""Скачивание и распаковка сенсорных данных NCLT.

Скрипт скачивает архив <date>_sen.tar.gz с сервера NCLT, распаковывает его
в папку data/ и проверяет наличие файла sen/ms25.csv (сырые данные IMU).
Если скачивание невозможно, выводится инструкция для ручного скачивания.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATE = "2013-01-10"
NCLT_HOSTS = [
    "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/sensor_data",
]
ARCHIVE_TEMPLATE = "{date}_sen.tar.gz"
REQUIRED_CSV = Path("sen") / "ms25.csv"


def archive_path(date: str) -> Path:
    return DATA_DIR / ARCHIVE_TEMPLATE.format(date=date)


def csv_path() -> Path:
    return DATA_DIR / REQUIRED_CSV


def build_urls(date: str) -> list[str]:
    name = ARCHIVE_TEMPLATE.format(date=date)
    return [f"{host}/{name}" for host in NCLT_HOSTS]


def manual_instructions(date: str) -> str:
    name = ARCHIVE_TEMPLATE.format(date=date)
    urls = "\n".join(f"  - {u}" for u in build_urls(date))
    return (
        "\n" + "=" * 78 + "\n"
        "НЕ УДАЛОСЬ СКАЧАТЬ АРХИВ АВТОМАТИЧЕСКИ.\n"
        "\nВыполните следующие шаги вручную:\n"
        f"  1. Откройте сайт NCLT: https://robots.engin.umich.edu/nclt/\n"
        f"  2. Скачайте архив сенсорных данных за {date}: {name}\n"
        "     Прямые ссылки (одна из них):\n"
        f"{urls}\n"
        f"  3. Положите файл в папку: {DATA_DIR}\n"
        "  4. Повторно запустите: python download_data.py\n"
        "     Скрипт распакует уже скачанный архив.\n"
        + "=" * 78 + "\n"
    )


def download(date: str, force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = archive_path(date)
    if target.exists() and not force:
        print(f"[i] Архив уже существует: {target}")
        return target

    last_error: Exception | None = None
    for url in build_urls(date):
        print(f"[i] Скачивание: {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(
                target, "wb"
            ) as out:
                shutil.copyfileobj(response, out)
            print(f"[+] Сохранено: {target} ({target.stat().st_size / 1e6:.1f} MB)")
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            print(f"[!] Ошибка при скачивании {url}: {exc}")
            if target.exists():
                target.unlink()

    raise RuntimeError(
        f"Не удалось скачать {ARCHIVE_TEMPLATE.format(date=date)}. "
        f"Последняя ошибка: {last_error}"
    )


def extract(archive: Path) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"Архив не найден: {archive}")
    print(f"[i] Распаковка {archive.name} в {DATA_DIR} ...")
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            member_path = (DATA_DIR / member.name).resolve()
            if not str(member_path).startswith(str(DATA_DIR.resolve())):
                raise RuntimeError(f"Небезопасный путь в архиве: {member.name}")
            members.append(member)
        tar.extractall(DATA_DIR, members=members)

    candidates = [csv_path()] + sorted(DATA_DIR.rglob("ms25.csv"))
    for candidate in candidates:
        if candidate.exists():
            if candidate != csv_path():
                csv_path().parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, csv_path())
            print(f"[+] Найден файл сырых данных: {csv_path()}")
            return csv_path()

    raise FileNotFoundError(
        "ms25.csv не найден после распаковки. "
        f"Проверьте содержимое папки {DATA_DIR}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка данных NCLT")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Дата (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Скачать заново")
    args = parser.parse_args()

    # 1. Скачиваем данные датчиков (если нет)
    if not csv_path().exists() or args.force:
        try:
            archive = download(args.date, force=args.force)
            extract(archive)
        except Exception as exc:
            print(f"[!] Ошибка с сенсорами: {exc}")
            print(manual_instructions(args.date))
            return 1
    else:
        print(f"[i] Файл датчиков уже существует: {csv_path()}")

    # 2. Скачиваем Ground Truth (если нет)
    gt_file = DATA_DIR / f"{args.date}_groundtruth.csv"
    if not gt_file.exists() or args.force:
        gt_url = f"https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/groundtruth/{args.date}_groundtruth.csv"
        print(f"[i] Скачивание Ground Truth: {gt_url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(gt_url, gt_file)
            print(f"[+] Ground Truth сохранен: {gt_file}")
        except Exception as e:
            print(f"[!] Не удалось скачать Ground Truth: {e}")
    else:
        print(f"[i] Ground Truth уже существует: {gt_file}")

    print("[+] Готово. Данные готовы к анализу.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
