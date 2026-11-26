import { request } from './client.js';

export const api = {
  signIn: data => request('/auth/login', { method: 'POST', body: data }),
  profile: () => request('/profile'),
  datasets: () => request('/datasets'),
  dataset: id => request('/datasets/' + id),
  upload: data => request('/datasets', { method: 'POST', body: data }),
  runs: () => request('/runs'),
  run: id => request('/runs/' + id),
  submit: data => request('/runs', { method: 'POST', body: data }),
  review: (id, data) => request('/runs/' + id + '/review', { method: 'POST', body: data }),
  reviews: () => request('/reviews'),
  audit: () => request('/audit'),
  modelCenter: () => request('/model-center'),
  validations: () => request('/validations'),
  validation: id => request('/validations/' + id),
  evaluate: data => request('/validations', { method: 'POST', body: data }),
  adopt: (data, idempotencyKey) => request('/model-adoptions', {
    method: 'POST',
    body: { idempotency_key: idempotencyKey, ...data },
    headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {},
  }),
  download: path => request(path, { download: true }),
};
