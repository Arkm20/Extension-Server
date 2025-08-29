import requests
import json
import time
import os
import collections
import getpass
import re

# --- Dependency Check and Setup ---
try:
    import google.generativeai as genai
except ImportError:
    print("Error: The 'google-generativeai' library is not installed.")
    print("Please install it by running: pip install google-generativeai")
    exit()

# --- Configuration ---
JIKAN_API_BASE_URL = "https://api.jikan.moe/v4"
CONTENT_FILE = "content.json"
CONFIG_FILE = "config.ini"

# Jikan rate limiting
REQUEST_TIMESTAMPS = collections.deque()
REQUEST_LIMIT = 3  # 3 requests per second
TIME_WINDOW = 1    # 1 second

# --- API Key and Configuration Management ---
def get_api_key():
    """Gets the Google AI API key, prompting the user if not found."""
    # Check environment variable first
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("Loaded Google API Key from environment variable.")
        input("Press Enter to continue...")
        return api_key

    # Check config file next
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            api_key = config.get("GOOGLE_API_KEY")
            if api_key:
                print("Loaded Google API Key from config.ini.")
                input("Press Enter to continue...")
                return api_key

    # If not found, prompt the user
    print("\n--- Google AI API Key Required ---")
    print("To use the AI generation feature, you need a Google AI API Key.")
    print("You can get a free key from Google AI Studio.")
    print("The key will be stored locally in 'config.ini' so you don't have to enter it again.")
    
    api_key = getpass.getpass("Please enter your Google AI API Key: ")

    # Save the key to config.ini for future use
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"GOOGLE_API_KEY": api_key}, f)
    
    print("API Key saved to config.ini.")
    return api_key


# --- Jikan API Interaction with Rate Limiting ---
def jikan_api_request(endpoint, params=None):
    """
    Makes a rate-limited request to the Jikan API.
    Waits if the request limit has been reached in the last second.
    """
    global REQUEST_TIMESTAMPS
    
    while True:
        now = time.time()
        # Remove timestamps older than the time window
        while REQUEST_TIMESTAMPS and REQUEST_TIMESTAMPS[0] < now - TIME_WINDOW:
            REQUEST_TIMESTAMPS.popleft()
        
        if len(REQUEST_TIMESTAMPS) < REQUEST_LIMIT:
            break
        
        # Calculate sleep time to respect the rate limit
        sleep_time = (REQUEST_TIMESTAMPS[0] + TIME_WINDOW) - now + 0.05 # small buffer
        print(f"Jikan rate limit reached. Waiting for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    try:
        REQUEST_TIMESTAMPS.append(time.time())
        print(f"Making Jikan request to: {JIKAN_API_BASE_URL}{endpoint}")
        response = requests.get(f"{JIKAN_API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n--- Jikan API Error --- \n{e}\n------------------")
        return None

# --- Helper Functions ---
def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_choice(max_choice, allow_back=True):
    """Gets and validates a user's integer choice."""
    while True:
        try:
            prompt = "> "
            choice = input(prompt)
            if allow_back and choice.lower() == 'b':
                return 'b'
            choice = int(choice)
            if 1 <= choice <= max_choice:
                return choice
            else:
                print(f"Invalid choice. Please enter a number between 1 and {max_choice}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def format_anime_data(anime_obj):
    """Formats Jikan anime data into the structure needed for content.json."""
    return {
        "id": anime_obj['mal_id'],
        "name": anime_obj.get('title_english') or anime_obj.get('title'),
        "image": anime_obj['images']['jpg']['large_image_url']
    }

def search_and_select_anime():
    """Prompts user to search for an anime, displays results, and returns the selected one."""
    query = input("Enter search term (or 'b' to go back): ")
    if query.lower() == 'b':
        return None
        
    results = jikan_api_request("/anime", params={"q": query, "limit": 10})
    if not results or not results.get('data'):
        print("No results found.")
        input("Press Enter to continue...")
        return None

    clear_screen()
    print(f"--- Search Results for '{query}' ---")
    for i, anime in enumerate(results['data'], 1):
        print(f"[{i}] {anime.get('title_english') or anime.get('title')} ({anime.get('type', 'N/A')}, {anime.get('year', 'N/A')})")
    
    print("\n[b] Back to previous menu")
    
    print("\nSelect an anime to add:")
    choice = get_choice(len(results['data']))
    if choice == 'b':
        return None
        
    return results['data'][choice - 1]

# --- Management Logic ---
def manage_spotlight(data):
    """Handles logic for managing the spotlight section."""
    while True:
        clear_screen()
        print("--- Manage Spotlight Section ---")
        if not data['spotlight']:
            print("Spotlight is currently empty.")
        else:
            for i, item in enumerate(data['spotlight'], 1):
                print(f"[{i}] {item['name']} (ID: {item['id']})")
        
        print("\nOptions:")
        print("[1] Add an anime to Spotlight")
        print("[2] Remove an anime from Spotlight")
        print("[b] Back to Main Menu")
        
        choice = input("> ").lower()
        
        if choice == '1':
            anime_obj = search_and_select_anime()
            if anime_obj:
                if any(item['id'] == anime_obj['mal_id'] for item in data['spotlight']):
                    print(f"'{anime_obj['title']}' is already in the spotlight.")
                else:
                    formatted = format_anime_data(anime_obj)
                    data['spotlight'].append(formatted)
                    print(f"Added '{formatted['name']}' to spotlight.")
                input("Press Enter to continue...")
        
        elif choice == '2':
            if not data['spotlight']:
                print("Nothing to remove.")
                input("Press Enter to continue...")
                continue
            print("Enter the number of the anime to remove (or 'b' to cancel):")
            remove_choice = get_choice(len(data['spotlight']))
            if remove_choice != 'b':
                removed = data['spotlight'].pop(remove_choice - 1)
                print(f"Removed '{removed['name']}' from spotlight.")
                input("Press Enter to continue...")

        elif choice == 'b':
            return

def manage_sections(data):
    """Handles logic for managing horizontal sections."""
    while True:
        clear_screen()
        print("--- Manage Horizontal Sections ---")
        if not data['sections']:
            print("No sections created yet.")
        else:
            for i, section in enumerate(data['sections'], 1):
                print(f"[{i}] {section['title']} ({len(section['items'])} items)")
        
        print("\nOptions:")
        print("[1] Create a new section")
        print("[2] Edit an existing section")
        print("[3] Delete a section")
        print("[b] Back to Main Menu")
        
        choice = input("> ").lower()

        if choice == '1':
            title = input("Enter title for new section: ")
            data['sections'].append({"title": title, "items": []})
            print(f"Section '{title}' created.")
            input("Press Enter...")
        
        elif choice == '2':
            if not data['sections']:
                print("No sections to edit.")
                input("Press Enter...")
                continue
            for i, section in enumerate(data['sections'], 1):
                print(f"[{i}] {section['title']} ({len(section['items'])} items)")
            print("Select a section to edit:")
            edit_choice = get_choice(len(data['sections']))
            if edit_choice != 'b':
                edit_section_menu(data['sections'][edit_choice - 1])

        elif choice == '3':
            if not data['sections']:
                print("No sections to delete.")
                input("Press Enter...")
                continue
            print("Select a section to delete:")
            delete_choice = get_choice(len(data['sections']))
            if delete_choice != 'b':
                removed = data['sections'].pop(delete_choice - 1)
                print(f"Deleted section '{removed['title']}'.")
                input("Press Enter...")

        elif choice == 'b':
            return

def edit_section_menu(section):
    """Menu for editing a specific section."""
    while True:
        clear_screen()
        print(f"--- Editing Section: {section['title']} ---")
        if not section['items']:
            print("This section is empty.")
        else:
            for i, item in enumerate(section['items'], 1):
                print(f"  [{i}] {item['name']} (ID: {item['id']})")
        
        print("\nOptions:")
        print("[1] Add an anime to this section (Manual Search)")
        print("[2] Remove an anime from this section")
        print("[3] Auto-populate this section (from Jikan)")
        print("[4] Generate content with AI")
        print("[5] Rename this section")
        print("[b] Back to Sections Menu")
        
        choice = input("> ").lower()

        if choice == '1':
            anime_obj = search_and_select_anime()
            if anime_obj:
                if any(item['id'] == anime_obj['mal_id'] for item in section['items']):
                    print(f"'{anime_obj['title']}' is already in this section.")
                else:
                    formatted = format_anime_data(anime_obj)
                    section['items'].append(formatted)
                    print(f"Added '{formatted['name']}' to '{section['title']}'.")
                input("Press Enter...")

        elif choice == '2':
            if not section['items']:
                print("Nothing to remove.")
                input("Press Enter...")
                continue
            print("Enter the number of the anime to remove:")
            remove_choice = get_choice(len(section['items']))
            if remove_choice != 'b':
                removed = section['items'].pop(remove_choice - 1)
                print(f"Removed '{removed['name']}' from '{section['title']}'.")
                input("Press Enter...")
        
        elif choice == '3':
            auto_populate_section(section)
        
        elif choice == '4':
            generate_with_ai(section)

        elif choice == '5':
            new_title = input(f"Enter new title for '{section['title']}': ")
            section['title'] = new_title
            print("Section renamed.")
            input("Press Enter...")

        elif choice == 'b':
            return

def auto_populate_section(section):
    """Automatically populates a section from a Jikan endpoint."""
    clear_screen()
    print(f"--- Auto-Populate Section: {section['title']} ---")
    print("Select a category to populate from:")
    print("[1] Top Anime by Popularity")
    print("[2] Upcoming Season")
    print("[3] Top Airing Anime")
    print("[b] Cancel")

    choice = get_choice(3)
    if choice == 'b':
        return

    endpoint_map = {
        1: ("/top/anime", {"filter": "bypopularity", "limit": 15}),
        2: ("/seasons/upcoming", {"limit": 15}),
        3: ("/top/anime", {"filter": "airing", "limit": 15})
    }
    endpoint, params = endpoint_map[choice]

    results = jikan_api_request(endpoint, params=params)
    if not results or not results.get('data'):
        print("Could not fetch data for this category.")
        input("Press Enter...")
        return
    
    added_count = 0
    skipped_count = 0
    for anime_obj in results['data']:
        if not any(item['id'] == anime_obj['mal_id'] for item in section['items']):
            section['items'].append(format_anime_data(anime_obj))
            added_count += 1
        else:
            skipped_count += 1
    
    print(f"Added {added_count} new items and skipped {skipped_count} duplicates in '{section['title']}'.")
    input("Press Enter...")

def generate_with_ai(section):
    """Generates content for a section using Google's Generative AI."""
    clear_screen()
    print(f"--- AI Content Generation for: {section['title']} ---")
    
    # 1. Get and configure API Key
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"An error occurred while configuring the AI model: {e}")
        input("Press Enter to return.")
        return

    # 2. Get user prompt
    print("Describe the kind of anime you want to find.")
    print("Examples: 'top 10 classic sci-fi anime', 'underrated shows with great world-building', 'anime for beginners'")
    user_prompt = input("\nEnter your prompt: ")
    if not user_prompt:
        return

    # 3. Query the AI model with improved prompt
    print("\nAsking the AI for suggestions... this may take a moment.")
    full_prompt = f"""List exactly 10 anime that fit this description: '{user_prompt}'.

IMPORTANT: Follow this exact format for your response:
- Return ONLY the anime titles
- One title per line
- No numbers, bullets, dashes, or prefixes
- No descriptions or explanations
- No extra text before or after the list
- Use the most commonly known English or romanized title

Example format:
Attack on Titan
Death Note
Spirited Away

Your response for '{user_prompt}':"""
    
    try:
        response = model.generate_content(full_prompt)
        if not response.text:
            print("The AI returned an empty response. Please try again.")
            input("Press Enter to return.")
            return
            
        # Clean and parse the response more robustly
        ai_suggestions = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            # Clean each line of common prefixes and formatting
            cleaned = line.strip()
            # Remove common prefixes like "1.", "- ", "• ", etc.
            cleaned = re.sub(r'^[\d\.\-\•\*\s]+', '', cleaned)
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 1:  # Ensure it's not just whitespace or single character
                ai_suggestions.append(cleaned)
        
        # Limit to 10 suggestions max
        ai_suggestions = ai_suggestions[:10]
        
    except Exception as e:
        print(f"An error occurred while communicating with the AI: {e}")
        input("Press Enter to return.")
        return

    if not ai_suggestions:
        print("The AI didn't return any valid suggestions. Try a different prompt.")
        input("Press Enter to return.")
        return

    print(f"\nAI suggested {len(ai_suggestions)} anime titles.")
    
    # 4. Process suggestions: Search Jikan and get user confirmation
    print("\n--- Confirm AI Suggestions ---")
    print("For each suggestion, I will find the closest match on MyAnimeList.")
    print("Please confirm if the match is correct.")
    
    confirmed_anime = []
    for i, suggestion in enumerate(ai_suggestions, 1):
        print(f"\n[{i}/{len(ai_suggestions)}] Searching for: '{suggestion}'...")
        results = jikan_api_request("/anime", params={"q": suggestion, "limit": 3})
        
        if not results or not results.get('data'):
            print(f"--> Could not find any match for '{suggestion}'.")
            continue
        
        # Show top match but also alternatives if the first doesn't seem right
        match = results['data'][0]
        title = match.get('title_english') or match.get('title')
        
        print(f"--> Best match: '{title}' ({match.get('type', 'N/A')}, {match.get('year', 'N/A')})")
        
        # Show alternatives if available
        if len(results['data']) > 1:
            print("    Alternatives:")
            for j, alt in enumerate(results['data'][1:3], 2):
                alt_title = alt.get('title_english') or alt.get('title')
                print(f"    [{j}] {alt_title} ({alt.get('type', 'N/A')}, {alt.get('year', 'N/A')})")
        
        while True:
            if len(results['data']) > 1:
                choice = input("    Choose: [1] Use best match, [2-3] Use alternative, [s] Skip, [Enter] Use best match: ").strip().lower()
            else:
                choice = input("    [Enter] Add this anime, [s] Skip: ").strip().lower()
            
            if choice == '' or choice == '1':
                selected_match = results['data'][0]
                break
            elif choice == 's':
                selected_match = None
                break
            elif choice in ['2', '3'] and len(results['data']) > int(choice) - 1:
                selected_match = results['data'][int(choice) - 1]
                break
            else:
                print("    Invalid choice. Please try again.")
        
        if selected_match:
            formatted = format_anime_data(selected_match)
            # Avoid adding duplicates
            if any(a['id'] == formatted['id'] for a in confirmed_anime):
                print(f"--> Already added '{formatted['name']}'. Skipping.")
            else:
                confirmed_anime.append(formatted)
                print(f"--> Added '{formatted['name']}' to the list.")
        else:
            print(f"--> Skipped '{suggestion}'.")

    # 5. Final review and add to section
    if not confirmed_anime:
        print("\nNo new anime were confirmed. Returning to menu.")
        input("Press Enter...")
        return

    clear_screen()
    print("--- Final Review ---")
    print("The following new anime will be added to the section:")
    for item in confirmed_anime:
        print(f"- {item['name']}")

    final_confirm = input("\nAdd these items to the section? [Y/n]: ").lower()
    if final_confirm == '' or final_confirm == 'y':
        added_count = 0
        skipped_count = 0
        for anime in confirmed_anime:
            if not any(item['id'] == anime['id'] for item in section['items']):
                section['items'].append(anime)
                added_count += 1
            else:
                skipped_count += 1
        print(f"\nSuccessfully added {added_count} new anime.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} anime that were already in the section.")
    else:
        print("Operation cancelled. No changes were made.")
    
    input("Press Enter to continue...")



# --- Main Application ---
def main():
    """Main function to run the content manager."""
    data = None
    
    # Check if a content file exists and prompt the user.
    if os.path.exists(CONTENT_FILE):
        clear_screen()
        print("--- Welcome Back ---")
        print(f"Found existing content file: '{CONTENT_FILE}'")
        print("\nWhat would you like to do?")
        print("[1] Load the existing content")
        print("[2] Start from scratch (Warning: saving will overwrite the old file)")

        while data is None:
            choice = input("> ")
            if choice == '1':
                try:
                    with open(CONTENT_FILE, 'r') as f:
                        data = json.load(f)
                        # Ensure the basic structure exists, in case the file is malformed
                        if 'spotlight' not in data: data['spotlight'] = []
                        if 'sections' not in data: data['sections'] = []
                    print("Content loaded successfully.")
                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"Error: Could not read or parse '{CONTENT_FILE}'. Starting from scratch.")
                    data = {"spotlight": [], "sections": []}
            elif choice == '2':
                print("Starting with a blank slate.")
                data = {"spotlight": [], "sections": []}
            else:
                print("Invalid choice. Please enter 1 or 2.")
        input("Press Enter to continue...")
    else:
        # If no content file exists, start from scratch automatically.
        print(f"No '{CONTENT_FILE}' found. Starting with a blank slate.")
        data = {"spotlight": [], "sections": []}
        input("Press Enter to continue...")


    while True:
        clear_screen()
        print("--- Anime Content Manager ---")
        print(" (with AI-Powered Suggestions)")
        print("\nSelect an option:")
        print("[1] Manage Spotlight Section")
        print("[2] Manage Horizontal Sections")
        print("[3] Save and Exit")
        print("[4] Exit Without Saving")
        
        choice = input("> ")

        if choice == '1':
            manage_spotlight(data)
        elif choice == '2':
            manage_sections(data)
        elif choice == '3':
            with open(CONTENT_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Content saved to {CONTENT_FILE}.")
            break
        elif choice == '4':
            print("Exiting without saving changes.")
            break
        else:
            print("Invalid option. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()