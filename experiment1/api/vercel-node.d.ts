declare module '@vercel/node' {
  export interface VercelRequest {
    method?: string;
    body?: any;
    query?: Record<string, string | string[] | undefined>;
    headers?: Record<string, string | string[] | undefined>;
  }

  export interface VercelResponse {
    status(code: number): VercelResponse;
    json(body: any): VercelResponse;
  }
}
