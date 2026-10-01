"""Generate synthetic distractor tools (other MCP servers' worth) to test routing at larger pool sizes."""
import json
from pathlib import Path

SPEC = {
    "filesystem": ("reading, writing, searching and organising files and folders on disk", [
        ("fs_read_file", "Read the full contents of a text file from the local disk.", ["path"]),
        ("fs_write_file", "Create or overwrite a file on the local disk with the given content.", ["path", "content"]),
        ("fs_list_directory", "List the files and subdirectories inside a directory.", ["path"]),
        ("fs_search_files", "Search for files by name pattern under a directory.", ["root", "pattern"]),
        ("fs_move_file", "Move or rename a file or directory.", ["source", "destination"]),
        ("fs_delete_file", "Delete a file from the local disk.", ["path"]),
        ("fs_get_file_info", "Get size, permissions and modification time of a file.", ["path"]),
        ("fs_make_directory", "Create a new directory, including missing parents.", ["path"])]),
    "github": ("working with GitHub repositories, issues and pull requests", [
        ("gh_create_issue", "Create a new issue in a GitHub repository.", ["repo", "title", "body"]),
        ("gh_list_issues", "List open issues of a GitHub repository, optionally filtered by label.", ["repo", "label"]),
        ("gh_get_pull_request", "Get the details, diff and review status of a pull request.", ["repo", "number"]),
        ("gh_create_pull_request", "Open a pull request from a branch.", ["repo", "head", "base", "title"]),
        ("gh_merge_pull_request", "Merge an approved pull request.", ["repo", "number", "method"]),
        ("gh_search_code", "Search code across GitHub repositories.", ["query"]),
        ("gh_list_commits", "List the recent commits on a branch.", ["repo", "branch"]),
        ("gh_create_branch", "Create a new branch from a commit or another branch.", ["repo", "name", "from"]),
        ("gh_list_workflow_runs", "List recent GitHub Actions workflow runs and their status.", ["repo"]),
        ("gh_get_file_contents", "Read a file from a GitHub repository at a given ref.", ["repo", "path", "ref"])]),
    "calendar": ("scheduling meetings and managing calendar events", [
        ("cal_list_events", "List calendar events in a date range.", ["start", "end"]),
        ("cal_create_event", "Create a calendar event with attendees and a time slot.", ["title", "start", "end", "attendees"]),
        ("cal_update_event", "Change the time or details of an existing event.", ["event_id", "changes"]),
        ("cal_delete_event", "Cancel and delete a calendar event.", ["event_id"]),
        ("cal_find_free_slot", "Find a common free time slot for several people.", ["attendees", "duration_minutes"]),
        ("cal_set_reminder", "Add a reminder to a calendar event.", ["event_id", "minutes_before"])]),
    "email": ("reading, searching, drafting and sending email", [
        ("mail_search", "Search the mailbox for messages matching a query.", ["query", "folder"]),
        ("mail_read_message", "Read a single email message by id.", ["message_id"]),
        ("mail_send", "Send an email to one or more recipients.", ["to", "subject", "body"]),
        ("mail_create_draft", "Save a draft email without sending it.", ["to", "subject", "body"]),
        ("mail_reply", "Reply to an email thread.", ["message_id", "body"]),
        ("mail_label_message", "Apply or remove a label on a message.", ["message_id", "label"])]),
    "web": ("browsing the web, fetching pages and general web search", [
        ("web_search", "Search the public web and return result snippets.", ["query"]),
        ("web_fetch_page", "Download a web page and return its text.", ["url"]),
        ("web_screenshot", "Take a screenshot of a web page.", ["url"]),
        ("web_extract_links", "Extract all hyperlinks from a web page.", ["url"]),
        ("web_check_status", "Check whether a website is reachable and its HTTP status.", ["url"])]),
    "database": ("querying and administering SQL databases", [
        ("db_run_query", "Run a read-only SQL query and return rows.", ["sql"]),
        ("db_list_tables", "List the tables of a database schema.", ["schema"]),
        ("db_describe_table", "Show columns, types and indexes of a table.", ["table"]),
        ("db_insert_rows", "Insert rows into a table.", ["table", "rows"]),
        ("db_explain_query", "Show the execution plan of a SQL query.", ["sql"]),
        ("db_export_csv", "Export the result of a query to a CSV file.", ["sql", "path"])]),
    "chat": ("sending and reading messages in team chat channels (Slack-like)", [
        ("chat_post_message", "Post a message to a team chat channel.", ["channel", "text"]),
        ("chat_list_channels", "List the channels in the workspace.", []),
        ("chat_read_history", "Read recent messages of a channel.", ["channel", "limit"]),
        ("chat_search_messages", "Search chat messages across channels.", ["query"]),
        ("chat_add_reaction", "React to a chat message with an emoji.", ["channel", "timestamp", "emoji"]),
        ("chat_set_status", "Set my chat presence status text.", ["text"])]),
    "tickets": ("tracking work items in a ticket / project management system (Jira-like)", [
        ("tkt_create", "Create a new ticket in a project.", ["project", "summary", "description"]),
        ("tkt_search", "Search tickets with a JQL-like query.", ["query"]),
        ("tkt_get", "Get one ticket with comments and history.", ["key"]),
        ("tkt_transition", "Move a ticket to another workflow status.", ["key", "status"]),
        ("tkt_assign", "Assign a ticket to a user.", ["key", "assignee"]),
        ("tkt_add_comment", "Add a comment to a ticket.", ["key", "text"]),
        ("tkt_list_sprints", "List the sprints of a board.", ["board"]),
        ("tkt_log_work", "Log worked time on a ticket.", ["key", "minutes"])]),
    "cloud": ("operating cloud infrastructure and Kubernetes clusters", [
        ("k8s_list_pods", "List pods in a Kubernetes namespace with their status.", ["namespace"]),
        ("k8s_get_logs", "Fetch the logs of a pod container.", ["pod", "container"]),
        ("k8s_scale_deployment", "Scale a deployment to a number of replicas.", ["name", "replicas"]),
        ("k8s_apply_manifest", "Apply a YAML manifest to the cluster.", ["manifest"]),
        ("cloud_list_instances", "List virtual machine instances in a cloud project.", ["project"]),
        ("cloud_start_instance", "Start a stopped virtual machine.", ["instance"]),
        ("cloud_stop_instance", "Stop a running virtual machine.", ["instance"]),
        ("cloud_list_buckets", "List object storage buckets.", ["project"]),
        ("cloud_get_billing", "Get the current month's cloud spend by service.", ["project"]),
        ("cloud_rotate_secret", "Rotate a stored secret and return the new version id.", ["name"])]),
    "finance": ("stock quotes, currency conversion and personal finance", [
        ("fin_get_quote", "Get the latest stock price for a ticker symbol.", ["symbol"]),
        ("fin_convert_currency", "Convert an amount between two currencies at today's rate.", ["amount", "from", "to"]),
        ("fin_list_transactions", "List my bank transactions in a date range.", ["start", "end"]),
        ("fin_get_portfolio", "Show my investment portfolio holdings and value.", []),
        ("fin_compute_loan", "Compute the monthly payment of a loan.", ["principal", "rate", "years"])]),
    "maps": ("weather forecasts, maps, routes and places", [
        ("geo_weather_forecast", "Get the weather forecast for a city.", ["city", "days"]),
        ("geo_route", "Compute a driving or transit route between two places.", ["origin", "destination", "mode"]),
        ("geo_search_places", "Search for restaurants, shops or other places near a location.", ["query", "location"]),
        ("geo_geocode", "Convert an address into coordinates.", ["address"]),
        ("geo_timezone", "Get the current time in a city.", ["city"])]),
    "notes": ("personal notes, documents and to-do lists", [
        ("note_create", "Create a new note in the notebook.", ["title", "body"]),
        ("note_search", "Search my notes by keyword.", ["query"]),
        ("note_append", "Append text to an existing note.", ["note_id", "text"]),
        ("todo_add", "Add an item to my to-do list.", ["text", "due"]),
        ("todo_list", "List my open to-do items.", []),
        ("todo_complete", "Mark a to-do item as done.", ["id"])]),
    "media": ("music playback, image generation and speech", [
        ("media_play_music", "Play a song, album or playlist.", ["query"]),
        ("media_pause", "Pause the current playback.", []),
        ("media_generate_image", "Generate an image from a text prompt.", ["prompt", "size"]),
        ("media_transcribe_audio", "Transcribe an audio file to text.", ["path"]),
        ("media_text_to_speech", "Read a text aloud and save it as an audio file.", ["text", "voice"])]),
    "smarthome": ("controlling smart-home devices", [
        ("home_set_light", "Turn a light on or off or set its brightness.", ["room", "state", "brightness"]),
        ("home_set_thermostat", "Set the target temperature of a room.", ["room", "celsius"]),
        ("home_lock_door", "Lock or unlock a door.", ["door", "state"]),
        ("home_list_devices", "List all smart-home devices and their state.", []),
        ("home_run_scene", "Activate a saved scene such as 'movie night'.", ["scene"]),
        ("home_get_camera_snapshot", "Get the latest snapshot from a security camera.", ["camera"])]),
}

tools, groups = [], {}
for cat, (desc, items) in SPEC.items():
    groups[cat] = {"description": desc, "tools": [i[0] for i in items]}
    for name, d, params in items:
        tools.append({"type": "function", "function": {
            "name": name, "description": d,
            "parameters": {"type": "object", "properties": {p: {"type": "string", "description": "The %s." % p.replace("_", " ")} for p in params}, "required": params[:1]}}})

out = Path(__file__).parent / "data"
(out / "distractor_tools.json").write_text(json.dumps(tools, indent=1))
(out / "distractor_groups.json").write_text(json.dumps({"groups": groups}, indent=1))
print(len(tools), "tools,", len(groups), "groups")
