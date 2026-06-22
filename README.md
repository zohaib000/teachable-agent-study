# Teachable Agent — Deployment Guide

This is the RAG-based teachable agent described in Section 3.8 of the
dissertation. Students teach "Codey" programming concepts; Codey asks
clarifying questions and gives adaptive feedback, instead of answering
questions itself.

## What's inside
- `app.py` — the Streamlit app (the agent + chat UI)
- `content/programming_basics.md` — the knowledge base the agent retrieves
  from (variables, conditionals, loops, functions, debugging, recursion)
- `requirements.txt` — Python dependencies
- `session_logs.csv` — auto-created once people start chatting; logs every
  turn with a session ID (no names) for your technical appendix

## Testing locally first (recommended before deploying)

1. In the `teachable_agent` folder, create a file named exactly `.env`
   (copy `.env.example` and rename it, or create a new text file named
   `.env`). Open it and put your real key in this format:
   ```
   OPENAI_API_KEY=sk-your-real-key-here
   ```
   No quotes, no spaces around the `=`.

2. In Command Prompt:
   ```
   cd E:\teachable_agent
   pip install -r requirements.txt
   streamlit run app.py
   ```
   You don't need `set OPENAI_API_KEY=...` anymore — the app reads it
   automatically from your `.env` file.

3. It should open `http://localhost:8501` in your browser. Try teaching it
   something (e.g. explain what a for-loop does) and confirm it asks a
   follow-up question instead of just agreeing.

**Important:** never upload your `.env` file to GitHub — it has your real
key in it. When you deploy to Streamlit Cloud (next section), you'll enter
the key in their separate "Secrets" box instead, not via this file.

## Fastest way to deploy (free, ~10 minutes)

1. **Get an OpenAI API key** (if you don't have one): https://platform.openai.com/api-keys
   Add a few dollars of credit — this whole study will cost well under $10
   in API usage for ~12 participants.

2. **Push this folder to a GitHub repo** (create a free GitHub account if
   needed):
   - Create a new repo (e.g. `teachable-agent-study`)
   - Upload these 4 items: `app.py`, `requirements.txt`, the `content/`
     folder, and this README.

3. **Deploy on Streamlit Community Cloud** (free): https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app" → select your repo → set main file to `app.py`
   - Before clicking deploy, open **"Advanced settings" → "Secrets"** and
     paste:
     ```
     OPENAI_API_KEY = "sk-your-real-key-here"
     ```
   - Click Deploy. After ~2 minutes you'll get a public link like:
     `https://your-app-name.streamlit.app`

4. **Test it yourself first** — open the link, try teaching it something
   (e.g. "a for loop repeats code a set number of times..."), confirm it
   asks you a follow-up question instead of just agreeing. If it works,
   you're ready to send the link to participants.

## Sending it to participants
Share, in this order:
1. The pre-survey link (your existing Google Form)
2. The agent link (this app) — ask them to spend 5-10 minutes teaching it
   at least one concept (loop, conditional, function, or a debugging
   example)
3. The post-survey link (same Google Form, completed again right after)

Ask each participant to note their **Session ID** (shown in the agent's
sidebar) — this lets us match their chat transcript to their survey
responses later for the analysis, without collecting their name.

## If something breaks
- "No OPENAI_API_KEY found" → the key wasn't added correctly in Secrets;
  re-check step 3.
- App loads but errors when you send a message → check the API key has
  available credit at platform.openai.com/usage.
- Anything else → send me a screenshot of the error and I'll fix it fast.

## Retrieving the data afterward
The "Download session logs" button is now **admin-only** — participants
won't see it. To access it yourself, open your deployed app's URL with
`?admin=1` added at the end, for example:
```
https://your-app-name.streamlit.app/?admin=1
```
That will reveal a "Researcher tools" section in the sidebar with the
download button. Send participants the plain link (without `?admin=1`).

Once participants have used it, the `session_logs.csv` file (inside the
Streamlit Cloud app's file system) will contain every conversation turn.
Streamlit Cloud apps can have ephemeral storage, so **download this file
periodically** using the admin link above — don't wait until the very end
to grab it.

**Before sending the real link to participants:** make sure you do a final
test run yourself first (using the plain link, not `?admin=1`, to see what
they'll see), and if a `session_logs.csv` already has your own test
conversations in it, delete it via Streamlit Cloud's file browser (or
redeploy fresh) so test data doesn't mix with real participant data.
