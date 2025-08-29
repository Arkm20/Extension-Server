import os
import importlib.util
import json
import socket
import threading
import uvicorn
import time
import asyncio
import uuid
from urllib.parse import urlparse, urljoin
from contextlib import asynccontextmanager
import re
import shutil
import subprocess
import signal
import traceback
import zipfile
import io
from fastapi import FastAPI, HTTPException, Query, Body, status, UploadFile, File, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import httpx
from fastapi.responses import StreamingResponse, JSONResponse, Response
from zeroconf import ServiceInfo, Zeroconf
from fpdf import FPDF
from PIL import Image
import math

# --- Module-specific imports ---
import requests
from bs4 import BeautifulSoup
import mangakatana
from selenium import webdriver
from selenium.common.exceptions import WebDriverException, JavascriptException, TimeoutException


# --- Global Event for Animation Control ---
server_ready_event = threading.Event()

def animate_loading(stop_event: threading.Event):
    """Displays a loading animation in the console until the stop_event is set."""
    animation_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
    idx = 0
    print("🎬 Starting Animex Extension Server...", end="", flush=True)
    while not stop_event.is_set():
        char = animation_chars[idx % len(animation_chars)]
        print(f" {char}", end="\r", flush=True)
        idx += 1
        time.sleep(0.08)
    time.sleep(2)
    print(" " * 5, end="\r", flush=True)
    print("📺 Animex Extension Server is ready for anime streaming!")
    print("Press CTRL+C to stop the Animex server\n")

# --- Utility Functions ---
def natural_sort_key(s):
    """A key function for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# --- Zeroconf Service Registration ---
zeroconf = Zeroconf()
service_info = None

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def register_service():
    global service_info
    try:
        host_ip = get_local_ip()
        host_name = socket.gethostname()
        port = 7275
        service_info = ServiceInfo(
            "_http._tcp.local.",
            f"Animex Extension API @ {host_name}._http._tcp.local.",
            addresses=[socket.inet_aton(host_ip)],
            port=port,
            properties={'app': 'animex-extension-api'},
            server=f"{host_name}.local.",
        )
        print(f"Registering service '{service_info.name}' on {host_ip}:{port}")
        zeroconf.register_service(service_info)
        print("Service registration completed")
    except Exception as e:
        print(f"Failed to register Zeroconf service: {e}")

async def unregister_service():
    if service_info:
        print(f"Unregistering service '{service_info.name}'")
        zeroconf.close()

# --- Hardcoded 9anime Module ---
NINEANIME_BASE_URL = "https://9anime.org.lv"
NINEANIME_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": NINEANIME_BASE_URL
}

def _9anime_get_anime_title_sync(jikan_id: int) -> Optional[str]:
    url = f"https://api.jikan.moe/v4/anime/{jikan_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        title = data.get("title_english") or data.get("title")
        return title or None
    except requests.exceptions.RequestException as e:
        print(f"9anime-Module: Jikan API request failed: {e}")
        return None

def _9anime_slugify_title(title: str) -> str:
    s = re.sub(r"(?i)([a-z0-9])'s\b", r"\1s", title).replace("’", "").replace("'", "")
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")

def _9anime_extract_nonce_from_html(html: str) -> Optional[str]:
    match = re.search(r"nonce\s*[:=]\s*['\"]([a-fA-F0-9]{10,})['\"]", html)
    return match.group(1) if match else None

def _9anime_resolve_final_link_with_selenium(initial_url: str) -> Optional[str]:
    driver = None
    print("9anime-Module: Initializing headless browser...")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--log-level=3")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(8)
        print(f"9anime-Module: Navigating to initial URL: {initial_url}")
        try:
            driver.get(initial_url)
        except TimeoutException:
            print("9anime-Module: Page load timed out as expected. Proceeding...")
            pass
        except WebDriverException as e:
            print(f"9anime-Module: Error during initial navigation: {e}")
            return None
        driver.switch_to.new_window('tab')
        driver.get("chrome://downloads")
        start_time = time.time()
        timeout = 15
        get_url_script = "return document.querySelector('downloads-manager').shadowRoot.querySelector('#downloadsList downloads-item').shadowRoot.querySelector('a#url').href;"
        print("9anime-Module: Resolving final download link...")
        while time.time() - start_time < timeout:
            try:
                final_url = driver.execute_script(get_url_script)
                if final_url and final_url != "about:blank":
                    print("9anime-Module: Final link resolved successfully.")
                    return final_url
            except JavascriptException:
                pass
            time.sleep(0.25)
        print(f"9anime-Module: Timed out after {timeout} seconds waiting for download link.")
        return None
    except Exception as e:
        print(f"9anime-Module: An unexpected error occurred in Selenium: {e}")
        return None
    finally:
        if driver:
            print("9anime-Module: Closing browser.")
            driver.quit()

async def get_9anime_iframe_source(mal_id: int, episode: int, dub: bool) -> Optional[str]:
    print(f"9anime-Module: Fetching iframe source for MAL ID {mal_id}, Ep {episode}, Dub: {dub}")
    async with httpx.AsyncClient(headers=NINEANIME_HEADERS, follow_redirects=True) as client:
        url = f"https://api.jikan.moe/v4/anime/{mal_id}"
        try:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            anime_title = data.get("title_english") or data.get("title")
            if not anime_title: return None
        except httpx.RequestError:
            return None
        slug = _9anime_slugify_title(anime_title)
        suffix = f"-dub-episode-{episode}" if dub else f"-episode-{episode}"
        watch_url = f"{NINEANIME_BASE_URL}/{slug}{suffix}"
        try:
            resp = await client.get(watch_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            embed_div = soup.find("div", id="embed_holder")
            iframe = embed_div.find("iframe", src=True) if embed_div else None
            return iframe["src"] if iframe else None
        except (httpx.HTTPStatusError, Exception):
            return None

def get_9anime_download_link(mal_id: int, episode: int, dub: bool, quality: str) -> Optional[str]:
    print(f"9anime-Module: Starting download process for MAL ID {mal_id}, Ep {episode}")
    anime_title = _9anime_get_anime_title_sync(mal_id)
    if not anime_title:
        print("9anime-Module: Failed to get anime title.")
        return None
    slug = f"{_9anime_slugify_title(anime_title)}{'-dub' if dub else ''}-episode-{episode}"
    watch_url = f"{NINEANIME_BASE_URL}/watch/{slug}"
    try:
        print(f"9anime-Module: Fetching nonce from {watch_url}")
        resp_page = requests.get(watch_url, headers=NINEANIME_HEADERS, timeout=10, allow_redirects=True)
        resp_page.raise_for_status()
        nonce = _9anime_extract_nonce_from_html(resp_page.text)
        if not nonce:
            print("9anime-Module: Could not extract nonce from page.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"9anime-Module: Failed to fetch watch page: {e}")
        return None
    ajax_url = f"{NINEANIME_BASE_URL}/wp-admin/admin-ajax.php"
    params = {"action": "fetch_download_links", "mal_id": mal_id, "ep": episode, "nonce": nonce}
    try:
        print(f"9anime-Module: Fetching download list from AJAX endpoint...")
        resp_ajax = requests.get(ajax_url, params=params, headers=NINEANIME_HEADERS, timeout=10)
        resp_ajax.raise_for_status()
        data = resp_ajax.json()
    except requests.exceptions.RequestException as e:
        print(f"9anime-Module: Failed to fetch AJAX data: {e}")
        return None
    if data.get("data", {}).get("status") != 200:
        print("9anime-Module: AJAX status was not successful.")
        return None
    soup = BeautifulSoup(data["data"]["result"], "html.parser")
    section_heading = soup.find("div", string=re.compile(r'\s*' + ("Dub" if dub else "Sub") + r'\s*'))
    if not section_heading:
        print("9anime-Module: Could not find Dub/Sub section.")
        return None
    links_container = section_heading.find_next_sibling("div")
    if not links_container:
        print("9anime-Module: Could not find links container.")
        return None
    quality_link_tag = links_container.find("a", string=lambda t: t and t.strip().lower() == quality.lower())
    initial_url = quality_link_tag['href'] if quality_link_tag else None
    if not initial_url:
        print(f"9anime-Module: Could not find link for quality '{quality}'.")
        return None
    return _9anime_resolve_final_link_with_selenium(initial_url)

# --- Hardcoded MangaKatana Module ---
async def _mangakatana_get_manga_titles_from_mal(mal_id: int, client: httpx.AsyncClient) -> List[str]:
    url = f"https://api.jikan.moe/v4/manga/{mal_id}"
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        titles = []
        if data.get("title"):
            titles.append(data["title"])
        if data.get("title_english") and data["title_english"] not in titles:
            titles.append(data["title_english"])
        for synonym in data.get("title_synonyms", []):
            if synonym not in titles:
                titles.append(synonym)
        if not titles:
            print(f"MangaKatana-Module: No titles found for MAL ID {mal_id}")
        return titles
    except httpx.RequestError as e:
        print(f"MangaKatana-Module: Jikan API request failed: {e}")
        return []
    except Exception as e:
        print(f"MangaKatana-Module: An unexpected error occurred during Jikan API call: {e}")
        return []

async def _mangakatana_scrape_images_from_url(url: str, client: httpx.AsyncClient) -> List[str]:
    try:
        print(f"MangaKatana-Module: Navigating to {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = await client.get(url, follow_redirects=True, headers=headers)
        response.raise_for_status()
        match = re.search(r'var thzq\s*=\s*\[(.*?),?\];', response.text)
        if not match:
            print("MangaKatana-Module: Could not find 'thzq' variable in the page source.")
            return []
        image_urls_str = match.group(1)
        image_links = [url.strip().strip("'\"") for url in image_urls_str.split(',') if url.strip()]
        print(f"MangaKatana-Module: Found {len(image_links)} images.")
        return image_links
    except httpx.RequestError as e:
        print(f"MangaKatana-Module: HTTP request failed: {e}")
        return []
    except Exception as e:
        print(f"MangaKatana-Module: An unexpected error occurred during scraping: {e}")
        return []

def _mangakatana_sync_get_chapters(titles: List[str]) -> Optional[List[Dict[str, Any]]]:
    for title in titles:
        try:
            print(f"MangaKatana-Module: Searching for '{title}' using the library...")
            results = mangakatana.search(title=title)
            if not results:
                print(f"MangaKatana-Module: Library found no results for '{title}'.")
                continue
            best_manga = None
            min_len_diff = float('inf')
            clean_search_title = re.sub(r'[^a-z0-9]', '', title.lower())
            for manga in results:
                clean_manga_title = re.sub(r'[^a-z0-9]', '', manga.title.lower())
                if clean_search_title in clean_manga_title:
                    len_diff = len(clean_manga_title) - len(clean_search_title)
                    if len_diff < min_len_diff:
                        min_len_diff = len_diff
                        best_manga = manga
            if best_manga is None:
                print(f"MangaKatana-Module: Could not find a confident match for '{title}'. Trying next title.")
                continue
            print(f"MangaKatana-Module: Selected '{best_manga.title}' as the best match for search term '{title}'.")
            print(f"MangaKatana-Module: Fetching chapters for '{best_manga.title}'...")
            chapters_from_lib = best_manga.chapter_list()
            formatted_chapters = []
            for ch in chapters_from_lib:
                chapter_number = "Unknown"
                try:
                    url_parts = urlparse(ch.url)
                    chapter_part = url_parts.path.strip('/').split('/')[-1]
                    chapter_number = re.sub(r'^c', '', chapter_part)
                except Exception as e:
                    print(f"MangaKatana-Module: Failed to extract chapter number from URL '{ch.url}': {e}. Falling back to title.")
                    num_search = re.search(r'(\d+(\.\d+)?)', ch.title)
                    if num_search:
                        chapter_number = num_search.group(1)
                    else:
                        print(f"MangaKatana-Module: Could not find chapter number in title '{ch.title}'.")
                        chapter_number = ch.title
                formatted_chapters.append({"title": ch.title, "url": ch.url, "chapter_number": str(chapter_number)})
            return formatted_chapters[::-1]
        except Exception as e:
            print(f"MangaKatana-Module: The 'mangakatana' library failed for title '{title}': {e}")
            continue
    print(f"MangaKatana-Module: Failed to find chapters for any of the titles: {titles}")
    return None

async def get_mangakatana_chapters(mal_id: int) -> Optional[List[Dict[str, Any]]]:
    async with httpx.AsyncClient() as client:
        manga_titles = await _mangakatana_get_manga_titles_from_mal(mal_id, client)
        if not manga_titles:
            return None
    loop = asyncio.get_running_loop()
    chapter_list = await loop.run_in_executor(None, _mangakatana_sync_get_chapters, manga_titles)
    return chapter_list

async def get_mangakatana_chapter_images(mal_id: int, chapter_num: str) -> Optional[List[str]]:
    all_chapters = await get_mangakatana_chapters(mal_id)
    if not all_chapters:
        return None
    target_chapter_url = None
    for chapter in all_chapters:
        if chapter.get("chapter_number") == str(chapter_num):
            target_chapter_url = chapter.get("url")
            print(f"MangaKatana-Module: Found URL for chapter {chapter_num}: {target_chapter_url}")
            break
    if not target_chapter_url:
        print(f"MangaKatana-Module: Could not find chapter {chapter_num} for MAL ID {mal_id}.")
        return None
    async with httpx.AsyncClient() as client:
        image_links = await _mangakatana_scrape_images_from_url(target_chapter_url, client)
    return image_links

# --- Extension Loading ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSIONS_DIR = os.path.join(APP_DIR, "extensions")
loaded_extensions = {}

def load_extensions(app: FastAPI):
    if not os.path.exists(EXTENSIONS_DIR):
        return
    for ext_name in os.listdir(EXTENSIONS_DIR):
        ext_path = os.path.join(EXTENSIONS_DIR, ext_name)
        if not os.path.isdir(ext_path):
            continue
        package_json_path = os.path.join(ext_path, "package.json")
        if not os.path.exists(package_json_path):
            continue
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                ext_meta = json.load(f)
            main_file = ext_meta.get("main", "extension.extn")
            ext_file_path = os.path.join(ext_path, main_file)
            module_name = f"extensions.{ext_name}.{main_file.split('.')[0]}"
            spec = importlib.util.spec_from_file_location(module_name, ext_file_path)
            ext_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ext_module)
            ext_module.EXT_PATH = os.path.abspath(ext_path)
            loaded_extensions[ext_name] = {"info": ext_meta, "instance": ext_module, "process": None, "server_url": None}
            if "port" in ext_meta and "start_command" in ext_meta:
                ext_port = ext_meta["port"]
                ext_start_command = ext_meta["start_command"]
                ext_server_url = f"http://127.0.0.1:{ext_port}"
                print(f"Starting extension '{ext_name}' server in the background.")
                process = subprocess.Popen(ext_start_command, shell=True, preexec_fn=os.setsid, cwd=ext_path)
                loaded_extensions[ext_name]["process"] = process
                loaded_extensions[ext_name]["server_url"] = ext_server_url
                print(f"Extension '{ext_name}' server process started with PID: {process.pid}.")
            if "static_folder" in ext_meta:
                static_path = os.path.join(ext_path, ext_meta["static_folder"])
                if os.path.isdir(static_path):
                    app.mount(f"/ext/{ext_name}/static", StaticFiles(directory=static_path), name=f"ext_{ext_name}_static")
                    print(f"Mounted static folder for '{ext_name}' at /ext/{ext_name}/static")
            print(f"Successfully loaded extension logic: {ext_meta.get('name', ext_name)}")
        except Exception as e:
            print(f"Failed to load extension {ext_name}: {e}")
            traceback.print_exc()

# --- Profile Management ---
class ProfileSettings(BaseModel):
    nsfw_enabled: bool = False
    module_preferences: Dict[str, List[str]] = Field(default_factory=dict)

class WatchedEpisode(BaseModel):
    watched_at: str
    season_number: int

class WatchHistoryItem(BaseModel):
    title: str
    last_watched: str
    episodes: Dict[str, WatchedEpisode] = Field(default_factory=dict)

class Profile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    avatar_url: str
    passcode: Optional[str] = None
    settings: ProfileSettings = Field(default_factory=ProfileSettings)
    watch_history: Dict[str, WatchHistoryItem] = Field(default_factory=dict)

class CreateProfileRequest(BaseModel):
    name: str
    passcode: Optional[str] = None

class LoginRequest(BaseModel):
    name: str
    passcode: str

profiles: Dict[str, Profile] = {}

# --- FastAPI App Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    load_extensions(app)
    loop = asyncio.get_event_loop()
    if not os.getenv("VERCEL"):
        loop.run_in_executor(None, register_service)
    server_ready_event.set()
    yield
    # --- Shutdown ---
    print("\nShutting down Animex Extension Server...")
    for ext_name, ext_data in loaded_extensions.items():
        process = ext_data.get("process")
        if process and process.poll() is None:
            print(f"Terminating extension '{ext_name}' server (PID: {process.pid})...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"Extension '{ext_name}' server terminated successfully.")
            except (ProcessLookupError, OSError) as e:
                print(f"Could not terminate extension '{ext_name}' server: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while terminating extension '{ext_name}': {e}")
    if not os.getenv("VERCEL"):
        await unregister_service()
    print("Animex Extension Server shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Animex Extensions API",
    description="A modular API for fetching anime stream and download information.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MangaDex API Configuration ---
MANGADEX_API_URL = "https://api.mangadex.org"

# --- Endpoints ---
@app.get("/settings/add-ons", response_model=List[Dict[str, Any]])
async def get_add_on_settings():
    settings = []
    for ext_name, ext_data in loaded_extensions.items():
        if "settings" in ext_data["info"]:
            settings.append({"extension_id": ext_name, **ext_data["info"]["settings"]})
    return settings

@app.get("/extensions", response_model=List[str])
async def get_extensions_list():
    return list(loaded_extensions.keys())

@app.post("/settings/add-on")
async def set_add_on_settings(settings_data: Dict[str, Any] = Body(...)):
    extension_id = settings_data.get("extension_id")
    if not extension_id or extension_id not in loaded_extensions:
        raise HTTPException(status_code=404, detail="Extension not found.")
    print(f"Received settings for {extension_id}: {settings_data}")
    return {"status": "success", "extension_id": extension_id, "settings": settings_data}

@app.get("/ext/{ext_name}/info", response_model=Dict[str, Any])
async def get_extension_info(ext_name: str):
    ext_data = loaded_extensions.get(ext_name)
    if not ext_data:
        raise HTTPException(status_code=404, detail="Extension not found.")
    return ext_data["info"]

@app.get("/identify", include_in_schema=False)
def identify_server():
    return {"app": "Animex Extension API", "version": "1.1.1"}

@app.get("/status")
async def get_status():
    return {"status": "online"}

@app.get("/export/series/{mal_id}")
async def export_series_package(mal_id: int, type: str = Query(..., enum=["anime", "manga"])):
    try:
        async with httpx.AsyncClient() as client:
            details_resp = await client.get(f"https://api.jikan.moe/v4/{type}/{mal_id}")
            details_resp.raise_for_status()
            series_data = details_resp.json().get("data", {})
            if not series_data:
                raise HTTPException(status_code=404, detail="Series not found on Jikan.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Jikan API error: {e.response.text}")
    series_title = series_data.get("title_english") or series_data.get("title", f"series_{mal_id}")
    if not series_title and type == "manga":
        series_title = series_data.get("title", f"series_{mal_id}")
    safe_series_title = "".join(c for c in series_title if c.isalnum() or c in (' ', '_')).rstrip()
    poster_url = series_data.get("images", {}).get("jpg", {}).get("large_image_url")
    poster_content = None
    if poster_url:
        try:
            async with httpx.AsyncClient() as client:
                poster_resp = await client.get(poster_url)
                poster_resp.raise_for_status()
                poster_content = poster_resp.content
        except httpx.RequestError:
            poster_content = None
    temp_dir = f"temp_export_{uuid.uuid4()}"
    series_dir = os.path.join(temp_dir, safe_series_title)
    os.makedirs(series_dir, exist_ok=True)
    try:
        meta_path = os.path.join(series_dir, "meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(series_data, f, indent=2)
        if poster_content:
            poster_path = os.path.join(series_dir, "poster.png")
            with open(poster_path, 'wb') as f:
                f.write(poster_content)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(series_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        zip_buffer.seek(0)
    finally:
        shutil.rmtree(temp_dir)
    zip_filename = f"{safe_series_title}.zip"
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=\"{zip_filename}\""})

@app.post("/profiles/{profile_id}/watch-history", status_code=status.HTTP_200_OK)
async def log_watch_history(profile_id: str, mal_id: int = Query(...), episode_number: int = Query(...), series_title: str = Query(...), season_number: int = Query(1)):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    profile = profiles[profile_id]
    mal_id_str = str(mal_id)
    episode_number_str = str(episode_number)
    now_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if mal_id_str not in profile.watch_history:
        profile.watch_history[mal_id_str] = WatchHistoryItem(title=series_title, last_watched=now_iso)
    profile.watch_history[mal_id_str].episodes[episode_number_str] = WatchedEpisode(watched_at=now_iso, season_number=season_number)
    return {"status": "success", "profile_id": profile_id, "mal_id": mal_id, "episode": episode_number}

class ProfileResponse(BaseModel):
    id: str
    name: str
    avatar_url: str
    has_passcode: bool
    settings: ProfileSettings
    watch_history: Dict[str, WatchHistoryItem]

@app.get("/profiles", response_model=List[ProfileResponse])
async def get_all_profiles():
    response_profiles = []
    for profile in profiles.values():
        response_profiles.append(ProfileResponse(id=profile.id, name=profile.name, avatar_url=profile.avatar_url, has_passcode=bool(profile.passcode), settings=profile.settings, watch_history=profile.watch_history))
    return sorted(response_profiles, key=lambda p: p.name.lower())

def get_profile_setting(user_id: str, setting_name: str) -> any:
    profile = profiles.get(user_id)
    if profile and hasattr(profile, 'settings'):
        return getattr(profile.settings, setting_name, None)
    return None

@app.post("/profiles", response_model=Profile, status_code=status.HTTP_201_CREATED)
async def create_profile(profile_req: CreateProfileRequest):
    if len(profiles) >= 10:
        raise HTTPException(status_code=400, detail="Maximum number of profiles reached.")
    if not profile_req.name or len(profile_req.name) > 20:
        raise HTTPException(status_code=400, detail="Profile name must be between 1 and 20 characters.")
    initial = profile_req.name[0].upper()
    avatar = f"https://placehold.co/100/FF9500/FFFFFF?text={initial}"
    new_profile = Profile(name=profile_req.name.strip(), avatar_url=avatar, passcode=profile_req.passcode if profile_req.passcode else None)
    profiles[new_profile.id] = new_profile
    return new_profile

@app.post("/login", response_model=Profile)
async def login_or_create_profile(login_req: LoginRequest):
    existing_profile = next((p for p in profiles.values() if p.name.lower() == login_req.name.lower()), None)
    if existing_profile:
        if existing_profile.passcode != login_req.passcode:
            raise HTTPException(status_code=401, detail="Invalid passcode.")
        return existing_profile
    else:
        if len(profiles) >= 10:
            raise HTTPException(status_code=400, detail="Maximum number of profiles reached.")
        if not login_req.name or len(login_req.name) > 20:
            raise HTTPException(status_code=400, detail="Profile name must be between 1 and 20 characters.")
        initial = login_req.name[0].upper()
        avatar = f"https://placehold.co/100/FF9500/FFFFFF?text={initial}"
        new_profile = Profile(name=login_req.name.strip(), avatar_url=avatar, passcode=login_req.passcode)
        profiles[new_profile.id] = new_profile
        return new_profile

@app.put("/profiles/{profile_id}", response_model=Profile)
async def update_profile(profile_id: str, updated_profile: Profile):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    if profile_id != updated_profile.id:
        raise HTTPException(status_code=400, detail="Profile ID mismatch.")
    profiles[profile_id] = updated_profile
    return updated_profile

@app.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_id: str):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    del profiles[profile_id]
    return

@app.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return profiles[profile_id]

@app.get("/admin/profiles/{passkey}", response_model=List[Dict[str, Any]])
async def view_all_profiles(passkey: str):
    ADMIN_PASSKEY = "supersecretpasskey"
    if passkey != ADMIN_PASSKEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Incorrect admin passkey.")
    return [{"name": p.name, "passcode": p.passcode, "id": p.id} for p in profiles.values()]

@app.get("/profiles/{profile_id}/watch-history", response_model=Dict[str, WatchHistoryItem])
async def get_watch_history(profile_id: str):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    profile = profiles.get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return getattr(profile, 'watch_history', {})

class ProfileLoginRequest(BaseModel):
    passcode: str

@app.post("/profiles/{profile_id}/login", status_code=status.HTTP_200_OK)
async def login_profile(profile_id: str, login_req: ProfileLoginRequest):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    profile = profiles[profile_id]
    if not profile.passcode:
        return {"status": "success", "message": "Profile has no passcode."}
    if profile.passcode == login_req.passcode:
        return {"status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Invalid passcode.")

@app.patch("/profiles/{profile_id}", response_model=Profile)
async def patch_profile(profile_id: str, profile_data: dict):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    current_profile = profiles[profile_id]
    if "name" in profile_data:
        current_profile.name = profile_data["name"].strip()
    if "settings" in profile_data:
        current_profile.settings = ProfileSettings(**profile_data["settings"])
    profiles[profile_id] = current_profile
    return current_profile

@app.patch("/profiles/{profile_id}/settings", response_model=Profile)
async def patch_profile_settings(profile_id: str, settings_data: ProfileSettings):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    current_profile = profiles[profile_id]
    current_profile.settings = settings_data
    profiles[profile_id] = current_profile
    return current_profile

@app.patch("/profiles/{profile_id}/module-preferences", response_model=Profile)
async def patch_module_preferences(profile_id: str, module_prefs: Dict[str, List[str]] = Body(...)):
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    current_profile = profiles[profile_id]
    current_profile.settings.module_preferences = module_prefs
    return current_profile

# --- Core Content Endpoints ---
def _get_cover_url_from_manga(manga: Dict[str, Any]) -> Optional[str]:
    cover_rel = next((rel for rel in manga.get("relationships", []) if rel.get("type") == "cover_art"), None)
    if cover_rel:
        file_name = cover_rel.get("attributes", {}).get("fileName")
        if file_name:
            return f"/mangadex/cover/{manga['id']}/{file_name}"
    return None

@app.get("/mangadex/search")
async def search_mangadex(q: str, profile_id: Optional[str] = Query(None)):
    nsfw_allowed = get_profile_setting(profile_id, 'nsfw_enabled')
    content_ratings = ["safe", "suggestive"]
    if nsfw_allowed:
        content_ratings.extend(["erotica", "pornographic"])
    params = {"title": q, "limit": 24, "contentRating[]": content_ratings, "includes[]": ["cover_art"]}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga", params=params)
            resp.raise_for_status()
            data = resp.json()
            for manga in data.get("data", []):
                manga["cover_url"] = _get_cover_url_from_manga(manga)
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/list")
async def list_mangadex(order: str = Query("latestUploadedChapter", enum=["latestUploadedChapter", "followedCount", "createdAt", "updatedAt"]), limit: int = Query(20, ge=1, le=100), profile_id: Optional[str] = Query(None)):
    nsfw_allowed = get_profile_setting(profile_id, 'nsfw_enabled')
    content_ratings = ["safe", "suggestive"]
    if nsfw_allowed:
        content_ratings.extend(["erotica", "pornographic"])
    params = {f"order[{order}]": "desc", "limit": limit, "contentRating[]": content_ratings, "includes[]": ["cover_art"]}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga", params=params)
            resp.raise_for_status()
            data = resp.json()
            for manga in data.get("data", []):
                manga["cover_url"] = _get_cover_url_from_manga(manga)
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/manga/{manga_id}")
async def get_mangadex_manga_details(manga_id: str):
    params = {"includes[]": ["cover_art", "author", "artist"]}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}", params=params)
            resp.raise_for_status()
            data = resp.json().get("data")
            if data:
                data["image_url"] = _get_cover_url_from_manga(data)
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/cover/{manga_id}/{file_name}")
async def get_mangadex_cover(manga_id: str, file_name: str, size: int = Query(256, enum=[256, 512])):
    cover_url = f"https://uploads.mangadex.org/covers/{manga_id}/{file_name}.{size}.jpg"
    headers = {"Referer": "https://mangadex.org/"}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(cover_url, headers=headers, timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(io.BytesIO(resp.content), media_type=content_type)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex cover API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching cover: {str(e)}")

@app.get("/mangadex/manga/{manga_id}/chapters")
async def get_mangadex_manga_chapters(manga_id: str, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    params = {"limit": limit, "offset": offset, "translatedLanguage[]": "en", "order[chapter]": "asc", "order[volume]": "asc", "includes[]": "scanlation_group", "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"]}
    headers = {"Referer": "https://mangadex.org/"}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}/feed", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return {"chapters": data.get("data", []), "total": data.get("total", 0)}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/manga/{manga_id}/all-chapters")
async def get_all_mangadex_manga_chapters(manga_id: str):
    all_chapter_ids = []
    limit = 100
    offset = 0
    params = {"limit": limit, "translatedLanguage[]": "en", "order[chapter]": "asc", "order[volume]": "asc", "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"]}
    headers = {"Referer": "https://mangadex.org/"}
    async with httpx.AsyncClient() as client:
        while True:
            try:
                params["offset"] = offset
                resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}/feed", params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                chapters_on_page = data.get("data", [])
                if not chapters_on_page:
                    break
                for chapter in chapters_on_page:
                    all_chapter_ids.append(chapter['id'])
                total = data.get("total", 0)
                offset += limit
                if offset >= total:
                    break
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error during pagination: {e.response.text}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred during pagination: {str(e)}")
    return {"chapter_ids": all_chapter_ids}

@app.get("/mangadex/manga/{manga_id}/chapter-nav-details/{chapter_id}")
async def get_mangadex_chapter_nav_details(manga_id: str, chapter_id: str):
    all_chapters = []
    limit = 500
    offset = 0
    total = 0
    params = {"limit": limit, "offset": offset, "translatedLanguage[]": "en", "order[chapter]": "asc", "order[volume]": "asc", "includes[]": "scanlation_group", "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"]}
    headers = {"Referer": "https://mangadex.org/"}
    async with httpx.AsyncClient() as client:
        try:
            while True:
                params["offset"] = offset
                resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}/feed", params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                chapters_on_page = data.get("data", [])
                all_chapters.extend(chapters_on_page)
                total = data.get("total", 0)
                offset += limit
                if offset >= total or not chapters_on_page:
                    break
            current_chapter_index = -1
            for i, chap in enumerate(all_chapters):
                if chap["id"] == chapter_id:
                    current_chapter_index = i
                    break
            if current_chapter_index == -1:
                raise HTTPException(status_code=404, detail="Chapter not found in manga feed.")
            current_chapter_details = all_chapters[current_chapter_index]
            next_chapter_id = None
            if current_chapter_index + 1 < len(all_chapters):
                next_chapter_id = all_chapters[current_chapter_index + 1]["id"]
            return {"current_chapter": current_chapter_details, "next_chapter_id": next_chapter_id, "total_chapters": total}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/chapter/{chapter_id}")
async def get_mangadex_chapter_images(chapter_id: str):
    at_home_url = f"{MANGADEX_API_URL}/at-home/server/{chapter_id}"
    headers = {"Referer": "https://mangadex.org/"}
    async with httpx.AsyncClient() as client:
        try:
            print(f"Fetching images for chapter {chapter_id} from MangaDex... URL: {at_home_url}")
            server_resp = await client.get(at_home_url, headers=headers, timeout=20)
            server_resp.raise_for_status()
            server_data = server_resp.json()
            base_url = server_data.get("baseUrl")
            chapter_hash = server_data.get("chapter", {}).get("hash")
            page_filenames = server_data.get("chapter", {}).get("data", [])
            if not all([base_url, chapter_hash, page_filenames]):
                print(f"Error: Incomplete data from MangaDex for chapter {chapter_id}. Data: {server_data}")
                raise HTTPException(status_code=500, detail="Incomplete data from MangaDex server endpoint")
            sorted_filenames = sorted(page_filenames, key=natural_sort_key)
            parsed_url = urlparse(base_url)
            server_host = parsed_url.netloc
            image_urls = [f"/mangadex/proxy/{server_host}/data/{chapter_hash}/{filename}" for filename in sorted_filenames]
            return image_urls
        except httpx.HTTPStatusError as e:
            print(f"MangaDex API returned an error for {at_home_url}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from MangaDex for {at_home_url}. Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse response from MangaDex.")
        except Exception as e:
            print(f"An unexpected error occurred in get_mangadex_chapter_images for chapter {chapter_id}:")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/proxy/{server_host}/data/{chapter_hash}/{filename:path}")
async def proxy_mangadex_image(server_host: str, chapter_hash: str, filename: str):
    if not server_host.endswith("mangadex.network"):
        raise HTTPException(status_code=400, detail="Invalid server host for MangaDex proxy.")
    image_url = f"https://{server_host}/data/{chapter_hash}/{filename}"
    headers = {"Referer": "https://mangadex.org/"}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(image_url, headers=headers, timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(io.BytesIO(resp.content), media_type=content_type)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex image proxy error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while proxying image: {str(e)}")

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/chapters/{mal_id}")
async def get_manga_chapters(mal_id: int, profile_id: Optional[str] = Query(None, description="ID of the active user profile")):
    current_profile = profiles.get(profile_id)
    nsfw_allowed = current_profile and current_profile.settings.nsfw_enabled
    # This endpoint now only uses MangaKatana, so NSFW check is based on its (non-existent) property
    print(f"Attempting to fetch chapters from MangaKatana for MAL ID: {mal_id}")
    try:
        chapters = await get_mangakatana_chapters(mal_id)
        if chapters:
            print(f"Success! Got {len(chapters)} chapters from MangaKatana")
            return {"chapters": chapters, "source_module": "mangakatana"}
    except Exception as e:
        print(f"MangaKatana failed with an error: {e}")
    raise HTTPException(status_code=404, detail="Could not retrieve chapters from MangaKatana.")

@app.get("/retrieve/{mal_id}/{chapter_num}")
async def get_manga_chapter_images(mal_id: int, chapter_num: str, profile_id: Optional[str] = Query(None, description="ID of the active user profile"), ext: Optional[str] = Query(None, description="The ID of the extension to use")):
    if ext:
        if ext in loaded_extensions:
            ext_data = loaded_extensions[ext]
            ext_info = ext_data.get("info", {})
            is_nsfw_ext = ext_info.get("nsfw", False)
            current_profile = profiles.get(profile_id)
            nsfw_allowed = current_profile and current_profile.settings.nsfw_enabled
            if is_nsfw_ext and not nsfw_allowed:
                raise HTTPException(status_code=403, detail=f"NSFW extension '{ext}' is disabled for this profile.")
            try:
                images_func = getattr(ext_data["instance"], "get_chapter_images", None)
                if images_func:
                    print(f"Attempting to fetch chapter images from extension: {ext}")
                    images = await images_func(mal_id, chapter_num)
                    if images is not None:
                        print(f"Success! Got {len(images)} images from extension {ext}")
                        return images
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Extension {ext} failed: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"Extension '{ext}' not found.")
    
    # --- Handle Hardcoded MangaKatana Request ---
    print(f"Attempting to fetch chapter images from MangaKatana for MAL ID {mal_id}, Chapter {chapter_num}")
    try:
        images = await get_mangakatana_chapter_images(mal_id, chapter_num)
        if images is not None:
            print(f"Success! Got {len(images)} images from MangaKatana")
            return images
    except Exception as e:
        print(f"MangaKatana failed with an error: {e}")
        traceback.print_exc()
                
    raise HTTPException(status_code=404, detail="Could not retrieve chapter images from any source.")

@app.get("/player/templates/pdf_reader.html", response_class=HTMLResponse)
async def serve_pdf_reader():
    try:
        with open("templates/pdf_reader.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF Reader UI not found.")
    
@app.get("/map/file/animekai", response_class=JSONResponse)
async def serve_animekai_map():
    try:
        with open("templates/map.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="AnimeKai Map UI not found.")
        
@app.get("/read/{source}/{manga_id}/{chapter_id}", response_class=HTMLResponse)
async def read_manga_chapter_source(source: str, manga_id: str, chapter_id: str):
    try:
        with open("templates/reader.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Reader UI not found.")

@app.get("/download-manga/site/{source}/{manga_id}/{chapter_id}", response_class=HTMLResponse)
async def download_manga_page(source: str, manga_id: str, chapter_id: str):
    try:
        with open("templates/download.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Download UI not found.")

@app.get("/download-manga/direct/{source}/{manga_id}/{chapter_id}")
async def download_manga_chapter_as_pdf(source: str, manga_id: str, chapter_id: str, request: Request):
    images = []
    manga_title = "Manga"
    chapter_num_str = chapter_id
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    if source == "mangadex":
        images = await get_mangadex_chapter_images(chapter_id)
        if images:
            try:
                details = await get_mangadex_manga_details(manga_id)
                manga_title = details.get("attributes", {}).get("title", {}).get("en", "MangaDex Manga")
            except Exception:
                manga_title = "MangaDex Manga"
    else:
        images = await get_mangakatana_chapter_images(mal_id=int(manga_id), chapter_num=chapter_id)
        if images:
            try:
                async with httpx.AsyncClient() as client:
                    jikan_url = f"https://api.jikan.moe/v4/manga/{manga_id}"
                    resp = await client.get(jikan_url)
                    resp.raise_for_status()
                    manga_title = resp.json().get("data", {}).get("title", "Manga")
            except Exception:
                manga_title = "Manga"
    if not images:
        raise HTTPException(status_code=404, detail="Could not retrieve chapter images.")
    print(f"Downloading {len(images)} images concurrently...")
    async def fetch_image_content(client, image_url, base_url):
        try:
            full_image_url = image_url if image_url.startswith('http') else f"{base_url}{image_url}"
            response = await client.get(full_image_url, timeout=60)
            response.raise_for_status()
            return (image_url, response.content)
        except Exception as e:
            print(f"Failed to download image {image_url}: {e}")
            return (image_url, None)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=10)) as client:
        tasks = [fetch_image_content(client, image_url, base_url) for image_url in images]
        image_results = await asyncio.gather(*tasks)
    print("All images downloaded. Stitching them together...")
    processed_images = []
    total_height = 0
    standard_width = 595
    for url, content in image_results:
        if content:
            try:
                with Image.open(io.BytesIO(content)) as img:
                    img = img.convert("RGB")
                    aspect_ratio = img.height / img.width
                    new_height = int(standard_width * aspect_ratio)
                    resized_img = img.resize((standard_width, new_height), Image.Resampling.LANCZOS)
                    processed_images.append(resized_img)
                    total_height += new_height
            except Exception as e:
                print(f"Could not process image {url}: {e}")
    if not processed_images:
        raise HTTPException(status_code=500, detail="No images could be processed.")
    composite_image = Image.new('RGB', (standard_width, total_height))
    current_y = 0
    for img in processed_images:
        composite_image.paste(img, (0, current_y))
        current_y += img.height
    print("Slicing composite image into PDF pages...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False, margin=0)
    pdf.set_title(f"Chapter {chapter_num_str} - {manga_title}")
    page_height_pt = 842
    num_pages = math.ceil(total_height / page_height_pt)
    for i in range(num_pages):
        y_start = i * page_height_pt
        box = (0, y_start, standard_width, y_start + page_height_pt)
        page_image = composite_image.crop(box)
        with io.BytesIO() as page_buffer:
            page_image.save(page_buffer, format="PNG")
            page_buffer.seek(0)
            pdf.add_page()
            pdf.image(page_buffer, x=0, y=0, w=pdf.w)
            print(f"Added page {i+1}/{num_pages} to PDF.")
    pdf_output = pdf.output(dest='S')
    safe_title = "".join([c for c in manga_title if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
    filename = f"{safe_title} - Chapter {chapter_num_str}.pdf"
    return Response(content=bytes(pdf_output), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})

@app.get("/proxy/browse", response_class=HTMLResponse)
async def proxy_browse(url: str):
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is required.")
    
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": base_url
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            
            content_type = resp.headers.get('content-type', '')
            if 'text/html' in content_type:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Rewrite links and sources to go through the proxy
                for tag in soup.find_all(href=True):
                    tag['href'] = f"/proxy/browse?url={urljoin(base_url, tag['href'])}"
                for tag in soup.find_all(src=True):
                    tag['src'] = f"/proxy/browse?url={urljoin(base_url, tag['src'])}"
                
                base = soup.find('base')
                if not base:
                    base = soup.new_tag('base', href=base_url)
                    if soup.head:
                        soup.head.insert(0, base)
                return HTMLResponse(content=str(soup))
            else:
                return Response(content=resp.content, media_type=content_type)

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstream error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while proxying: {str(e)}")

@app.get("/proxy/iframe", response_class=HTMLResponse)
async def proxy_iframe(url: str):
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is required.")
    
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": base_url
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            
            content_type = resp.headers.get('content-type', '')
            if 'text/html' in content_type:
                soup = BeautifulSoup(resp.text, "html.parser")
                base = soup.find('base')
                if not base:
                    base = soup.new_tag('base', href=base_url)
                    if soup.head:
                        soup.head.insert(0, base)
                return HTMLResponse(content=str(soup))
            else:
                return Response(content=resp.content, media_type=content_type)

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstream error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while proxying iframe: {str(e)}")


@app.get("/iframe-src")
async def get_iframe_source(mal_id: int = Query(...), episode: int = Query(...), dub: bool = Query(False)):
    print(f"Attempting to fetch iframe source from 9anime for MAL ID: {mal_id}")
    try:
        iframe_src = await get_9anime_iframe_source(mal_id, episode, dub)
        if iframe_src:
            print(f"Success! Got source from 9anime: {iframe_src}")
            return {"src": iframe_src, "source_module": "9anime"}
    except Exception as e:
        print(f"9anime failed with an error: {e}")
    raise HTTPException(status_code=404, detail="Could not retrieve an iframe source from 9anime.")

@app.get("/download")
async def get_download_link(id: int = Query(...), episode: int = Query(...), dub: bool = Query(False), quality: str = Query("720p")):
    print(f"Attempting to fetch download link from 9anime for MAL ID: {id}")
    try:
        loop = asyncio.get_running_loop()
        download_link = await loop.run_in_executor(None, get_9anime_download_link, id, episode, dub, quality)
        if download_link:
            print(f"Success! Got download link from 9anime: {download_link}")
            return {"download_link": download_link, "source_module": "9anime"}
    except Exception as e:
        print(f"9anime failed with a download error: {e}")
        traceback.print_exc()
    raise HTTPException(status_code=404, detail="Could not retrieve a download link from 9anime.")

@app.get("/map/mal/{mal_id}")
async def mal_to_kitsu(mal_id: int):
    ANIME_DB_URL = "https://raw.githubusercontent.com/Fribb/anime-lists/refs/heads/master/anime-offline-database-reduced.json"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(ANIME_DB_URL, timeout=15)
            resp.raise_for_status()
            anime_data_cache = resp.json()
        for anime in anime_data_cache:
            if anime.get("mal_id") == mal_id:
                kitsu_id = anime.get("kitsu_id")
                if kitsu_id:
                    return {"kitsu_id": kitsu_id}
                raise HTTPException(status_code=404, detail=f"No Kitsu ID for MAL ID {mal_id}")
        raise HTTPException(status_code=404, detail=f"MAL ID {mal_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load anime mapping data: {str(e)}")
    
ANILIST_API_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($malId: Int) {
  Media(idMal: $malId, type: ANIME) {
    id
    bannerImage
    coverImage {
      extraLarge
      large
      medium
    }
  }
}
"""
@app.get("/anime/image")
async def get_anime_image(mal_id: int = Query(..., description="MyAnimeList anime ID"), cover: bool = Query(False, description="Return cover image instead of banner")):
    payload = {"query": ANILIST_QUERY, "variables": {"malId": mal_id}}
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(ANILIST_API_URL, json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
            media = data.get("data", {}).get("Media")
            if not media:
                raise HTTPException(status_code=404, detail="Anime not found on AniList")
            if cover:
                image_url = (media.get("coverImage", {}).get("extraLarge") or media.get("coverImage", {}).get("large") or media.get("coverImage", {}).get("medium"))
            else:
                image_url = media.get("bannerImage")
            if not image_url:
                raise HTTPException(status_code=404, detail="Image not available for this anime")
            img_response = await client.get(image_url, timeout=15)
            img_response.raise_for_status()
            content_type = img_response.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(io.BytesIO(img_response.content), media_type=content_type)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from upstream API: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.api_route("/ext/{ext_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_to_extension(ext_name: str, path: str, request: Request):
    ext_data = loaded_extensions.get(ext_name)
    if not ext_data or not ext_data.get("server_url"):
        raise HTTPException(status_code=404, detail=f"No running server found for extension '{ext_name}'.")
    target_url = ext_data["server_url"]
    target_url_full = f"{target_url}/{path}"
    print(f"Proxying request for '{ext_name}' to: {target_url_full}?{request.query_params}")
    async with httpx.AsyncClient() as client:
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "accept-encoding"]}
            response = await client.request(method=request.method, url=target_url_full, headers=headers, content=await request.body(), params=request.query_params, timeout=30.0)
            response_headers = {k:v for k,v in response.headers.items() if k.lower() != 'content-encoding'}
            return Response(content=response.content, status_code=response.status_code, headers=response_headers)
        except httpx.HTTPStatusError as e:
            print(f"Proxy HTTPStatusError: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Extension proxy error: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Proxy RequestError: {e}")
            raise HTTPException(status_code=502, detail=f"Bad Gateway: Cannot connect to extension server for '{ext_name}'.")
        except Exception as e:
            print(f"Proxy Unexpected Error: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during proxying: {e}")

# --- Static Files Hosting ---
app.mount("/proxy", StaticFiles(directory="proxy", html=True), name="proxy")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/", StaticFiles(directory="animex", html=True), name="static_site")
print("Static files mounted at /data and /")

# --- To make the server runnable directly ---
if __name__ == "__main__":
    animation_thread = threading.Thread(target=animate_loading, args=(server_ready_event,), daemon=True)
    animation_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=7275, log_level="info")