import { put } from '@vercel/blob';
import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { sessionId, csv } = req.body;

    if (!csv || typeof csv !== 'string') {
      return res.status(400).json({ error: 'Missing csv data' });
    }

    const timestamp = Date.now();
    const filename = `data_${sessionId || 'anonymous'}_${timestamp}.csv`;

    const blob = await put(filename, csv, {
      access: 'public',
      contentType: 'text/csv',
    });

    return res.status(200).json({ success: true, url: blob.url });
  } catch (error) {
    console.error('Upload error:', error);
    return res.status(500).json({ error: 'Upload failed' });
  }
}
