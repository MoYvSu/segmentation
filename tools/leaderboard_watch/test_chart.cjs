// 只使用虚构分数验证尺度、时间范围和断线处理。
const test = require('node:test');
const assert = require('node:assert/strict');
const FX = require('./web/chart-core.js');
const point = (t,value) => ({t,value,observedAt:t,synthetic:false});

test('0.0001 分的变化占据可读的纵轴范围', () => {
  const s=FX.scale([82.37,82.3701]);
  assert.ok(s.max-s.min<.001);
  assert.ok(.0001/(s.max-s.min)>.2);
  assert.ok(s.decimals>=5);
  assert.ok(new Set(s.ticks.map((n)=>n.toFixed(s.decimals))).size===s.ticks.length);
});
test('放大后自动轴排除窗口外的高分异常点', () => {
  const points=[point(0,99),point(300000,82.37),point(600000,82.3701)];
  const visible=FX.clip(points,{from:300000,to:600000},750000);
  const s=FX.scale(visible.map((p)=>p.value));
  assert.ok(s.max<82.371);
});
test('常数数列有稳定尺度，分数不被修改', () => {
  const s=FX.scale([82.37,82.37]);
  assert.ok(s.max>s.min&&s.min<82.37&&s.max>82.37);
  assert.ok(s.max-s.min<.002);
});
test('不同分数层级可按区间变化公平比较', () => {
  const a=FX.relative([point(0,82.37),point(300000,82.3701)],'change','score');
  const b=FX.relative([point(0,20),point(300000,20.0002)],'change','score');
  assert.equal(a.points[0].value,0);assert.equal(b.points[0].value,0);
  assert.ok(Math.abs(a.points[1].value-.0001)<1e-10);
  assert.ok(b.points[1].value>a.points[1].value);
});
test('名次提高、距榜首缩小在变化模式中为正', () => {
  assert.equal(FX.relative([point(0,3),point(1,1)],'change','rank').points[1].value,2);
  assert.equal(FX.relative([point(0,2),point(1,1)],'change','gap').points[1].value,1);
});
test('零基数百分比保持未知，不能变成 Infinity 或 0', () => {
  const result=FX.relative([point(0,0),point(1,1)],'percent','score');
  assert.deepEqual(result.points.map((p)=>p.value),[null,null]);
});
test('时间滚轮缩放保留鼠标锚点', () => {
  const view={from:100000,to:500000},anchor=250000;
  const next=FX.zoom(view,{from:0,to:1000000},.5,anchor);
  assert.equal(next.to-next.from,200000);
  assert.equal((anchor-view.from)/(view.to-view.from),(anchor-next.from)/(next.to-next.from));
});
test('连续放大有最小范围，平移不能越过历史两端', () => {
  const bounds={from:0,to:40*86400000};
  let view=FX.preset(bounds,3600000);
  for(let i=0;i<100;i++)view=FX.zoom(view,bounds,.5);
  assert.ok(view.to-view.from>=60000-1e-5);
  const left=FX.pan(view,bounds,-1e20),right=FX.pan(view,bounds,1e20);
  assert.equal(left.from,bounds.from);assert.equal(right.to,bounds.to);
  assert.ok(Math.abs((left.to-left.from)-(view.to-view.from))<1e-3);
});
test('时间快捷范围只取已有历史，不补造过去', () => {
  const bounds={from:500000,to:1000000};
  assert.deepEqual(FX.preset(bounds,86400000),bounds);
  assert.deepEqual(FX.preset(bounds,60000),{from:940000,to:1000000});
});
test('只在相邻有效检查之间延续阶梯值', () => {
  const result=FX.clip([point(0,1),point(300000,2)],{from:60000,to:240000},750000);
  assert.deepEqual(result.map((p)=>[p.t,p.value,p.synthetic]),[[60000,1,true],[240000,1,true]]);
  assert.equal(FX.summary(result).observations,0);
});
test('采集失败和长时间休眠不补线', () => {
  const gap=FX.clip([point(0,1),point(300000,null),point(600000,2)],{from:60000,to:240000},750000);
  const sleep=FX.clip([point(0,1),point(86400000,2)],{from:60000,to:240000},750000);
  assert.equal(gap.length,0);assert.equal(sleep.length,0);
});
test('名次绝对轴不产生负数或小数刻度', () => {
  const s=FX.scale([1,1,2],{integer:true,floor:1});
  assert.ok(s.min>=1);assert.ok(s.ticks.every(Number.isInteger));
});
test('手动纵轴保持指定范围，缩放能放大微小变化', () => {
  const s=FX.scale([0,1],{manual:{min:.1,max:.2}});
  assert.equal(s.min,.1);assert.equal(s.max,.2);
  const next=FX.zoomScale(s,.1,.15);
  assert.ok(Math.abs((next.max-next.min)-.01)<1e-12);
});
test('失败检查不拿上一快照伪装成有效观测', () => {
  const history={observations:[{checked_at:'2026-10-26T10:00:00+08:00',success:1,snapshot_id:1},{checked_at:'2026-10-26T10:05:00+08:00',success:0,snapshot_id:1}],snapshots:[{id:1,rows:[{team_id:'DEMO',score:'9.2',rank:1}]}]};
  const values=FX.series(history,['DEMO'],'score').get('DEMO');
  assert.equal(values[0].value,9.2);assert.equal(values[1].value,null);
});
