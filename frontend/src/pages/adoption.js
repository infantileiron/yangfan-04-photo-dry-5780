import { api } from '../api/workspace.js';
import { empty, heading } from '../components/common/heading.js';
import { table } from '../components/common/table.js';
import { bindForm, notice } from '../hooks/forms.js';
import { navigate } from '../router/routes.js';
import { escapeHtml as e, timeText } from '../utils/format.js';

const pct = value => value === null || value === undefined ? '—' : (Number(value) * 100).toFixed(1) + '%';
const seconds = value => value === null || value === undefined ? '—' : Number(value).toFixed(2) + ' 秒';

function conditionRows(report) {
  return table([
    { key: 'label', label: '提前划定工况' },
    { key: 'coverage', label: '样本 / 试机日' },
    { key: 'old_wet_rate', label: '旧版未干率' },
    { key: 'new_wet_rate', label: '新版未干率' },
    { key: 'wet_rate_delta', label: '差值（新−旧）' },
    { key: 'upper_bound', label: '95% 单侧上限' },
    { key: 'evidence', label: '证据判断' },
  ], report.conditions, (row, column, value) => {
    if (column.key === 'coverage') return `${row.sample_count} 张 / ${row.date_count} 天`;
    if (column.key === 'old_wet_rate' || column.key === 'new_wet_rate') return e(pct(value));
    if (column.key === 'wet_rate_delta' || column.key === 'upper_bound') {
      const cls = value > 0.000000001 ? 'risk-high' : 'risk-ok';
      return `<span class="${cls}">${e(pct(value))}</span>`;
    }
    if (column.key === 'evidence') {
      return `<span class="${row.safe ? 'risk-ok' : 'risk-high'}">${row.safe ? '✓ ' : '✕ '}${e(value)}</span>`;
    }
    return e(value);
  });
}

function sampleRows(report) {
  const rows = report.rows.slice(0, 20).map(row => ({
    ...row,
    old_state: row.old_wet ? '未干' : '已干',
    new_state: row.new_wet ? '未干' : '已干',
  }));
  return table([
    { key: 'sample_id', label: '试纸' },
    { key: 'trial_date', label: '试机日' },
    { key: 'condition_label', label: '工况' },
    { key: 'measured_s', label: '实测秒数' },
    { key: 'old_seconds', label: '旧计时' },
    { key: 'new_seconds', label: '新计时' },
    { key: 'old_state', label: '旧版结果' },
    { key: 'new_state', label: '新版结果' },
  ], rows, (row, column, value) => {
    if (column.key === 'old_state' || column.key === 'new_state') {
      const wet = column.key === 'old_state' ? row.old_wet : row.new_wet;
      return `<span class="${wet ? 'risk-high' : 'risk-ok'}">${wet ? '未干/蹭花风险' : '计时时已干'}</span>`;
    }
    if (typeof value === 'number') return e(Number(value).toFixed(1));
    return e(value ?? '—');
  });
}

export async function adoptionPage(profile, account, id) {
  const [center, datasets] = await Promise.all([api.modelCenter(), api.datasets()]);
  const active = center.active_model;
  const selected = id ? await api.validation(id) : null;
  const report = selected?.report || null;
  const adoption = selected && center.adoptions.find(item => item.validation_id === selected.id);
  const stale = selected && (
    selected.base_model_key !== active.key ||
    (selected.base_fingerprint && active.fingerprint && selected.base_fingerprint !== active.fingerprint) ||
    (selected.base_state_adoption_id || null) !== (active.adoption_id || null)
  );
  const blockingStale = Boolean(stale && !adoption);
  const canManage = account.role === 'analyst' || account.role === 'reviewer';
  const candidateDefault = center.models.find(model => model.key === 'swift-cross-v2')?.key
    || center.models.find(model => model.key !== active.key)?.key
    || active.key;

  const comparisonList = center.validations.map(item => `<a class="version ${selected?.id === item.id ? 'current' : ''}" href="#adoption/${item.id}">
    <strong>${e(item.label || '留验比较')}</strong>
    <span class="${item.eligible ? 'risk-ok' : 'risk-high'}">${item.eligible ? '可换用' : '不可换用'}</span>
    <span>${e(timeText(item.created_at))}</span></a>`).join('') || '<p>暂无比较单</p>';

  const form = canManage ? `<form id="validation-form">
    <label>历史训练 / 选型 / 校准数据<select name="training_dataset_id" required>${datasets.map(item =>
      `<option value="${item.id}">${e(item.name)} · ${item.count} 行</option>`).join('')}</select></label>
    <label>另导的留验数据（可不选）<select name="holdout_dataset_id"><option value="">留验行已在上方同一 CSV 中</option>${datasets.map(item =>
      `<option value="${item.id}">${e(item.name)} · ${item.count} 行</option>`).join('')}</select></label>
    <label>比较单名称<input name="label" maxlength="100" value="留验比较" required></label>
    <label>候选新模型<select name="candidate_model_key">${center.models.map(model =>
      `<option value="${e(model.key)}" ${model.key === candidateDefault ? 'selected' : ''}>${e(model.name)}</option>`).join('')}</select></label>
    <details class="details"><summary>覆盖率参数（通常留空）</summary>
      <label>共同校准覆盖率（%，留空用模型默认）<input name="parameter" type="number" min="50" max="95" step="1" placeholder="默认"></label>
      <label>候选模型覆盖率（%，优先于共同值）<input name="candidate_parameter" type="number" min="50" max="95" step="1" placeholder="例如 60"></label>
    </details>
    <p class="muted">数据中请将带首次耐擦实测秒数的行标为「留验试纸」；这些日期不会进入新旧模型训练、选型和校准。</p>
    <div class="form-error" data-error role="alert"></div>
    <button type="submit">生成留验比较</button>
  </form>` : '<p class="muted">分析员可导入留验数据并发起比较；其他角色可查看依据。</p>';

  const gate = selected ? `<div class="gate-banner ${adoption || (selected.eligible && !blockingStale) ? 'gate-ok' : 'gate-block'}">
    <h2>${adoption ? '该比较已完成换用' : blockingStale ? '在用模型已变化，需重新比较' : selected.eligible ? '证据满足换用门槛' : '证据不足或存在工况未干率上升'}</h2>
    <p>${e(report?.summary || '')}</p>
    ${blockingStale ? '<p><strong>在用模型变了要重比：</strong>本单基线不是当前在用模型，不能据此换用。</p>' : ''}
    ${selected.eligible && !blockingStale && !adoption && canManage ? '<button type="button" id="adopt-model" class="primary">确认换用候选模型</button><p class="muted">换用失败保留旧版；重试同一个请求也只生效一次。</p>' : ''}
    ${adoption ? `<p class="muted">换用单 ${e(adoption.id.slice(0, 8))} 已生效；重复请求不会再次生效。</p>` : ''}
    ${!canManage ? '<p class="muted">仅分析员或复核员可执行换用。</p>' : ''}
  </div>` : '';

  const overview = report ? `<dl class="metrics">
    <div><dt>留验试纸 / 试机日</dt><dd>${report.overall.sample_count} 张 / ${report.overall.date_count} 天</dd></div>
    <div><dt>旧版平均计时</dt><dd>${seconds(report.overall.old_mean_seconds)}</dd></div>
    <div><dt>新版平均计时</dt><dd>${seconds(report.overall.new_mean_seconds)}</dd></div>
    <div><dt>整体省时</dt><dd class="${report.overall.time_saved ? 'risk-ok' : 'risk-high'}">${seconds(report.overall.mean_saving_seconds)}（${pct(report.overall.saving_pct)}）</dd></div>
    <div><dt>旧 / 新未干率</dt><dd>${pct(report.overall.old_wet_rate)} / ${pct(report.overall.new_wet_rate)}</dd></div>
    <div><dt>省时的按日 95% 区间</dt><dd>${seconds(report.overall.saving_ci_low)} ～ ${seconds(report.overall.saving_ci_high)}</dd></div>
  </dl>` : '';

  return {
    html: heading(profile.labels.adoption || '模型换用', profile.model_validation?.rule || '') +
      `<div class="board"><aside class="side">
        <h2>在用模型</h2><p><strong>${e(active.name)}</strong></p><p class="muted">${e(active.key)}</p>
        <p class="muted">换用时照常接单；排队试算永久沿用下单时快照的模型与覆盖率。</p>
        <h2>比较单</h2>${comparisonList}
        <h2>发起新比较</h2>${form}
      </aside>
      <section class="work">
        ${gate}
        ${selected && report ? `
          <div class="work-head"><div><h2>${e(selected.label)}</h2>
          <p class="muted">旧：${e(report.base_model.name)} · 新：${e(report.candidate_model.name)} · ${e(timeText(selected.created_at))}</p></div></div>
          ${overview}
          <h3>分工况未干率证据</h3>
          ${conditionRows(report)}
          <h3>同批试纸计时与未干判定（前 20 张）</h3>
          ${sampleRows(report)}
          <details class="details" open><summary>页面依据与统计口径</summary><ul>${(report.basis || []).map(note => '<li>' + e(note) + '</li>').join('')}</ul></details>
          <details class="details"><summary>按试机日汇总</summary>${table([
            {key:'trial_date',label:'试机日'},{key:'sample_count',label:'张数'},
            {key:'old_wet_rate',label:'旧未干率'},{key:'new_wet_rate',label:'新未干率'},{key:'saving_seconds',label:'省时'},
          ], report.daily, (row,column,value)=> column.key.includes('rate') ? e(pct(value)) : (typeof value === 'number' ? e(value.toFixed(2)) : e(value)))}</details>
        ` : empty('尚未选择留验比较', '导入含「留验试纸」用途和实测干燥秒数的 CSV，然后在左侧生成比较。', '')}
      </section></div>`,
    mount(root, refresh) {
      bindForm(root, 'validation-form', async data => {
        const numberOrNull = key => data.get(key) === '' ? null : Number(data.get(key));
        const created = await api.evaluate({
          training_dataset_id: data.get('training_dataset_id'),
          holdout_dataset_id: data.get('holdout_dataset_id') || null,
          label: data.get('label'),
          base_model_key: active.key,
          candidate_model_key: data.get('candidate_model_key'),
          parameter: numberOrNull('parameter'),
          candidate_parameter: numberOrNull('candidate_parameter'),
        });
        notice('留验比较已生成。');
        navigate('adoption', created.id);
      });
      const button = root.querySelector('#adopt-model');
      if (button) {
        button.onclick = async () => {
          button.disabled = true;
          try {
            await api.adopt({ validation_id: selected.id }, 'adoption-' + selected.id);
            notice('已换用新模型；同一请求重试不会重复生效。');
            refresh();
          } catch (error) {
            notice(error.message, 'error');
            button.disabled = false;
          }
        };
      }
    },
  };
}
