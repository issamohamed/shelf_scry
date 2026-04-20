import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const GROQ_API_KEY = process.env.GROQ_API_KEY;
if (!GROQ_API_KEY) {
  console.error('Error: GROQ_API_KEY is not set. Copy .env.example to .env and add your key.');
  process.exit(1);
}

const PORT = process.env.PORT || 3000;

const app = express();
app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

const upload = multer({ storage: multer.memoryStorage() });

const SYSTEM_PROMPT = `You are a book recommendation engine. The user will describe a book plot or theme in casual natural language. Your job is to return a JSON array of real books that match the description.

Return ONLY a valid JSON array with no markdown, no code fences, no explanation. Each item must have exactly these fields:
- "title": the exact book title
- "author": the author's full name

Return between 6 and 10 books. Only include books that genuinely exist and match the described plot. Prioritize well-known titles but include lesser-known ones if they are a strong match.

Example output format:
[
  { "title": "Throne of Glass", "author": "Sarah J. Maas" },
  { "title": "Assassin's Apprentice", "author": "Robin Hobb" }
]`;

async function getGroqBooks(query) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      temperature: 0.4,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: query },
      ],
    }),
  });

  if (!res.ok) {
    throw new Error(`Groq API error: ${res.status} ${res.statusText}`);
  }

  const data = await res.json();
  let text = data.choices[0].message.content;

  // Strip accidental markdown code fences
  text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();

  return JSON.parse(text);
}

async function getOpenLibraryData(title, author) {
  const params = new URLSearchParams({
    title,
    author,
    limit: '1',
    fields: 'key,title,author_name,subject,cover_i,first_sentence',
  });

  const searchRes = await fetch(`https://openlibrary.org/search.json?${params}`);
  if (!searchRes.ok) return null;

  const searchData = await searchRes.json();
  if (!searchData.docs || searchData.docs.length === 0) return null;

  const doc = searchData.docs[0];
  if (!doc.key) return null;

  const workId = doc.key.replace('/works/', '');
  const workUrl = `https://openlibrary.org/works/${workId}.json`;

  let synopsis = '';
  try {
    const workRes = await fetch(workUrl);
    if (workRes.ok) {
      const workData = await workRes.json();
      if (workData.description) {
        synopsis =
          typeof workData.description === 'string'
            ? workData.description
            : workData.description.value || '';
      }
    }
  } catch {
    // ignore — fall back to first_sentence below
  }

  // Fall back to first_sentence if synopsis is missing or too short
  if (synopsis.length < 40 && doc.first_sentence) {
    const fs = doc.first_sentence;
    synopsis = typeof fs === 'string' ? fs : fs.value || synopsis;
  }

  const result = {
    title: doc.title || title,
    author: (doc.author_name && doc.author_name[0]) || author,
    synopsis,
    source_url: `https://openlibrary.org/works/${workId}`,
  };

  if (doc.cover_i) {
    result.cover_url = `https://covers.openlibrary.org/b/id/${doc.cover_i}-L.jpg`;
  }

  if (doc.subject && doc.subject.length > 0) {
    result.genres = doc.subject.slice(0, 4);
  }

  return result;
}

const SURPRISE_SYSTEM_PROMPT = `You are a book recommendation engine that specializes in variety. Return a JSON array of exactly 10 books.

Rules:
- Span wildly different genres: literary fiction, thriller, sci-fi, fantasy, romance, non-fiction, historical, horror, comedy, etc.
- Mix time periods across different centuries and decades
- Mix fame levels: some well-known, some obscure, some cult classics
- No two books should feel similar in genre or tone
- Be genuinely unpredictable — avoid defaulting to the same canonical titles

Return ONLY a valid JSON array with no markdown, no code fences. Each item: { "title": "...", "author": "..." }`;

const BOTD_SYSTEM_PROMPT = `You are a literary curator. Pick exactly one book from any genre, era, or fame level. Be unpredictable — sometimes pick a pulpy thriller, sometimes a Nobel Prize winner, sometimes an obscure cult classic, sometimes a beloved children's classic. Never default to the same famous titles.

Return ONLY a valid JSON object with no markdown, no code fences:
{ "title": "...", "author": "...", "description": "3-5 sentences about the plot" }`;

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.post('/transcribe', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file provided' });
  }
  try {
    const form = new FormData();
    form.append('file', new Blob([req.file.buffer], { type: 'audio/webm' }), 'audio.webm');
    form.append('model', 'whisper-large-v3');

    const groqRes = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` },
      body: form,
    });

    if (!groqRes.ok) {
      throw new Error(`Groq Whisper error: ${groqRes.status} ${groqRes.statusText}`);
    }

    const data = await groqRes.json();
    res.json({ transcript: data.text });
  } catch (err) {
    console.error('Transcription error:', err);
    res.status(500).json({ error: 'Transcription failed' });
  }
});

app.post('/search', async (req, res) => {
  const { query } = req.body;

  if (!query || query.trim().length < 3) {
    return res.status(400).json({ error: 'Query must be at least 3 characters' });
  }

  let groqBooks;
  try {
    groqBooks = await getGroqBooks(query.trim());
  } catch (err) {
    console.error('Groq error:', err);
    if (err instanceof SyntaxError) {
      return res.status(500).json({ error: 'Failed to parse Groq response' });
    }
    return res.status(500).json({ error: 'Failed to get book recommendations' });
  }

  if (!Array.isArray(groqBooks) || groqBooks.length === 0) {
    return res.json({ results: [] });
  }

  let results;
  try {
    const settled = await Promise.all(
      groqBooks.map((book) => getOpenLibraryData(book.title, book.author).catch(() => null))
    );
    results = settled.filter(Boolean);
  } catch (err) {
    console.error('Open Library error:', err);
    return res.status(500).json({ error: 'Failed to fetch book metadata' });
  }

  res.json({ results });
});

app.get('/surprise', async (_req, res) => {
  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GROQ_API_KEY}` },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        temperature: 1.0,
        messages: [
          { role: 'system', content: SURPRISE_SYSTEM_PROMPT },
          { role: 'user', content: `Give me 10 varied books. Seed: ${Math.random()}` },
        ],
      }),
    });

    if (!groqRes.ok) throw new Error(`Groq error: ${groqRes.status}`);

    const data = await groqRes.json();
    let text = data.choices[0].message.content;
    text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
    const books = JSON.parse(text);

    const settled = await Promise.all(
      books.map((b) => getOpenLibraryData(b.title, b.author).catch(() => null))
    );
    res.json({ results: settled.filter(Boolean) });
  } catch (err) {
    console.error('Surprise error:', err);
    res.status(500).json({ error: 'Failed to fetch surprise books' });
  }
});

app.get('/book-of-the-day', async (_req, res) => {
  try {
    const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GROQ_API_KEY}` },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        temperature: 1.0,
        messages: [
          { role: 'system', content: BOTD_SYSTEM_PROMPT },
          { role: 'user', content: `Pick one book. Seed: ${Math.random()}` },
        ],
      }),
    });

    if (!groqRes.ok) throw new Error(`Groq error: ${groqRes.status}`);

    const data = await groqRes.json();
    let text = data.choices[0].message.content;
    text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
    res.json(JSON.parse(text));
  } catch (err) {
    console.error('Book of the day error:', err);
    res.status(500).json({ error: 'Failed to fetch book of the day' });
  }
});

const server = app.listen(PORT, () => {
  console.log(`Shelf Scry running at http://localhost:${PORT}`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Error: Port ${PORT} is already in use. Run: lsof -ti :${PORT} | xargs kill -9`);
    process.exit(1);
  }
  throw err;
});
