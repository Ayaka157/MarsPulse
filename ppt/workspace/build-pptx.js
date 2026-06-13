const pptxgen = require('pptxgenjs');
const html2pptx = require('C:/Users/53631/.claude/skills/pptx/scripts/html2pptx.js');
const path = require('path');

const OUT_DIR = 'C:/Users/53631/Desktop/比賽/航天/MarsPulse_提交版/output';
const WS_DIR = __dirname;

// [filename, width_px, height_px]
const IMG = {
  cover:  ['step5_dashboard.png',    3284, 2063],
  s3:     ['step1_trajectories.png', 2100, 1500],
  s4:     ['step2_agent_actions.png',2100, 1650],
  s5:     ['step2_comparison.png',   2700, 1200],
  s6:     ['step4_ablation.png',     2708, 1185],
  s7:     ['step3_sensitivity.png',  2700, 750],
  s8:     ['step5_dashboard.png',    3284, 2063],
};

function addImg(slide, ph, file, iw, ih, maxH) {
  const ar = iw / ih;
  let h = Math.min(maxH || ph.h, ph.h);
  let w = h * ar;
  if (w > ph.w) { w = ph.w; h = w / ar; }
  slide.addImage({
    path: path.join(OUT_DIR, file),
    x: ph.x + (ph.w - w) / 2,
    y: ph.y + (ph.h - h) / 2,
    w, h,
  });
}

async function main() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'MarsPulse Team';
  pptx.title = 'MarsPulse 答辩PPT';

  // S1 Cover + dashboard
  const s1 = await html2pptx(path.join(WS_DIR, 's1-cover.html'), pptx);
  if (s1.placeholders[0]) addImg(s1.slide, s1.placeholders[0], ...IMG.cover, 2.5);

  // S2 Background (no image)
  await html2pptx(path.join(WS_DIR, 's2-bg.html'), pptx);

  // S3 Scenario + trajectories
  const s3 = await html2pptx(path.join(WS_DIR, 's3-scenario.html'), pptx);
  if (s3.placeholders[0]) addImg(s3.slide, s3.placeholders[0], ...IMG.s3, 2.9);

  // S4 Method + agent actions
  const s4 = await html2pptx(path.join(WS_DIR, 's4-method.html'), pptx);
  if (s4.placeholders[0]) addImg(s4.slide, s4.placeholders[0], ...IMG.s4, 3.0);

  // S5 Results + comparison
  const s5 = await html2pptx(path.join(WS_DIR, 's5-results.html'), pptx);
  if (s5.placeholders[0]) addImg(s5.slide, s5.placeholders[0], ...IMG.s5, 3.1);

  // S6 Ablation + ablation chart
  const s6 = await html2pptx(path.join(WS_DIR, 's6-ablation.html'), pptx);
  if (s6.placeholders[0]) addImg(s6.slide, s6.placeholders[0], ...IMG.s6, 3.1);

  // S7 Findings + sensitivity
  const s7 = await html2pptx(path.join(WS_DIR, 's7-findings.html'), pptx);
  if (s7.placeholders[0]) addImg(s7.slide, s7.placeholders[0], ...IMG.s7, 3.1);

  // S8 Output + dashboard
  const s8 = await html2pptx(path.join(WS_DIR, 's8-output.html'), pptx);
  if (s8.placeholders[0]) addImg(s8.slide, s8.placeholders[0], ...IMG.s8, 2.5);

  // S9 Conclusion (no image)
  await html2pptx(path.join(WS_DIR, 's9-conclusion.html'), pptx);

  const outPath = 'C:/Users/53631/Desktop/比賽/航天/MarsPulse_提交版/ppt/MarsPulse_答辩PPT_带图.pptx';
  await pptx.writeFile({ fileName: outPath });
  console.log('Done: ' + outPath);
}

main().catch(err => { console.error(err); process.exit(1); });
