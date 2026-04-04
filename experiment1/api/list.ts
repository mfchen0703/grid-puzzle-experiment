import { list } from '@vercel/blob';
import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { blobs } = await list();

    const files = blobs.map((blob) => ({
      filename: blob.pathname,
      url: blob.url,
      uploadedAt: blob.uploadedAt,
      size: blob.size,
    }));

    return res.status(200).json({ files });
  } catch (error) {
    console.error('List error:', error);
    return res.status(500).json({ error: 'Failed to list files' });
  }
}
