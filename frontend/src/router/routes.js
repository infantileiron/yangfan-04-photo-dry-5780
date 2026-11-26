const pages = new Set(['datasets', 'runs', 'results', 'handoff', 'adoption']);
const aliases = {
  models: 'adoption',
  model: 'adoption',
  validations: 'adoption',
  validation: 'adoption',
  holdout: 'adoption',
  switch: 'adoption',
};

export function route() {
  const [rawPage, id] = location.hash.slice(1).split('/');
  const page = aliases[rawPage] ?? (pages.has(rawPage) ? rawPage : 'datasets');
  return { page, id: /^[a-f0-9-]{36}$/.test(id ?? '') ? id : '' };
}

export function navigate(page, id = '') {
  location.hash = '#' + page + (id ? '/' + id : '');
}
