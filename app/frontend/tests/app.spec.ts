import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import type { Example } from '../src/types'

const examples: Example[] = JSON.parse(readFileSync(new URL('../public/examples.json', import.meta.url), 'utf8'))
const recorded = examples[0].result

async function ready(page: Page) {
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ready', ready: true, busy: false } }))
  await page.route('**/api/models', route => route.fulfill({ json: {
    models: [{ backend: 'local', model: 'test-local', available: true, reason: null },
      { backend: 'api', model: 'test-api', available: true, reason: null }], default_backend: 'local',
  } }))
}

test('asks through the selected backend, opens a source, and resets the session', async ({ page }) => {
  await ready(page)
  await page.route('**/api/ask', async route => {
    expect(route.request().postDataJSON()).toEqual({ question: 'What is a database?', language: 'en', backend: 'api' })
    await route.fulfill({ json: { ...recorded, query: 'What is a database?', backend: 'api', language: 'en' } })
  })
  await page.goto('/')
  await expect(page.getByText('Sẵn sàng', { exact: true })).toBeVisible()
  await page.getByLabel('Mô hình trả lời').selectOption('api')
  await page.getByLabel('Ngôn ngữ trả lời').selectOption('en')
  await page.getByLabel('Câu hỏi của bạn', { exact: true }).fill('What is a database?')
  await page.getByRole('button', { name: 'Gửi câu hỏi', exact: true }).click()
  await expect(page.getByText(recorded.answer, { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Mở nguồn S1', exact: true }).click()
  await expect(page.getByText('Đoạn nguồn S1', { exact: true })).toBeVisible()
  await expect(page.getByText(/Trang PDF 35/)).toBeVisible()
  await page.screenshot({ path: 'test-results/desktop-answer.png', fullPage: true })
  await page.getByRole('button', { name: 'Xóa hội thoại' }).click()
  await expect(page.getByText(recorded.answer, { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('Câu hỏi của bạn', { exact: true })).toHaveValue('')
})

test('keeps the question after an error and prevents duplicate submissions while waiting', async ({ page }) => {
  await ready(page)
  let release!: () => void
  const waiting = new Promise<void>(resolve => { release = resolve })
  let calls = 0
  await page.route('**/api/ask', async route => {
    calls += 1
    await waiting
    await route.fulfill({ status: 429, json: { detail: 'Đang xử lý một câu hỏi khác. Vui lòng thử lại sau.' } })
  })
  await page.goto('/')
  await expect(page.getByText('Sẵn sàng', { exact: true })).toBeVisible()
  await page.getByLabel('Câu hỏi của bạn', { exact: true }).fill('Cơ sở dữ liệu là gì?')
  await page.getByRole('button', { name: 'Gửi câu hỏi', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Gửi câu hỏi', exact: true })).toBeDisabled()
  await expect(page.getByLabel('Mô hình trả lời')).toBeDisabled()
  await expect(page.getByLabel('Ngôn ngữ trả lời')).toBeDisabled()
  await expect(page.getByText('Đang trả lời…')).toBeVisible()
  release()
  await expect(page.getByRole('alert')).toContainText('Đang xử lý một câu hỏi khác')
  await expect(page.getByLabel('Câu hỏi của bạn', { exact: true })).toHaveValue('Cơ sở dữ liệu là gì?')
  expect(calls).toBe(1)
})

test('keeps the draft offline and opens sources on mobile after reconnecting', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.route('**/api/**', route => route.abort('connectionrefused'))
  await page.goto('/')
  await expect(page.getByText('Chưa kết nối', { exact: true })).toBeVisible()
  await page.getByLabel('Câu hỏi của bạn', { exact: true }).fill(recorded.query)
  await expect(page.getByRole('button', { name: 'Gửi câu hỏi', exact: true })).toBeDisabled()
  await ready(page)
  await page.route('**/api/ask', route => route.fulfill({ json: recorded }))
  await page.getByRole('button', { name: 'Thử lại', exact: true }).click()
  await expect(page.getByText('Sẵn sàng', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Câu hỏi của bạn', { exact: true })).toHaveValue(recorded.query)
  await page.getByRole('button', { name: 'Gửi câu hỏi', exact: true }).click()
  await expect(page.getByText(recorded.answer, { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Mở nguồn S1', exact: true }).click()
  await expect(page.getByText('Đoạn nguồn S1', { exact: true })).toBeVisible()
  await expect(page.getByText(/Trang PDF 35/)).toBeVisible()
  await page.screenshot({ path: 'test-results/mobile-source.png', fullPage: true })
  await page.getByRole('button', { name: 'Đóng nguồn', exact: true }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true })
})

test('shows an insufficient-evidence answer and its uncited context', async ({ page }) => {
  await ready(page)
  const insufficient = examples.find(value => value.result.status === 'insufficient_evidence')!.result
  await page.route('**/api/ask', route => route.fulfill({ json: insufficient }))
  await page.goto('/')
  await expect(page.getByText('Sẵn sàng', { exact: true })).toBeVisible()
  await page.getByLabel('Câu hỏi của bạn', { exact: true }).fill(insufficient.query)
  await page.getByRole('button', { name: 'Gửi câu hỏi', exact: true }).click()
  await expect(page.getByText('Chưa đủ bằng chứng', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Mở nguồn S1', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: /Xem ngữ cảnh/ }).click()
  await expect(page.getByText('Đoạn nguồn S1', { exact: true })).toBeVisible()
})

test('landing page has no console errors', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await ready(page)
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Hỏi đáp', exact: true })).toBeVisible()
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true })
  expect(errors).toEqual([])
})

test('live Local browser → FastAPI → saved Qdrant → Ollama', async ({ page }) => {
  test.skip(process.env.LIVE_RAG !== '1', 'Enable explicitly for a real model call')
  test.setTimeout(720_000)
  await page.goto('/')
  await expect(page.getByText('Sẵn sàng', { exact: true })).toBeVisible({ timeout: 120_000 })
  await page.getByLabel('Câu hỏi của bạn', { exact: true }).fill('Cơ sở dữ liệu là gì?')
  const response = page.waitForResponse(value => value.url().endsWith('/api/ask') && value.request().method() === 'POST', { timeout: 600_000 })
  await page.getByRole('button', { name: 'Gửi câu hỏi', exact: true }).click()
  const actual = await response
  expect(actual.status()).toBe(200)
  const result = await actual.json()
  expect(result.status).toBe('answered')
  expect(result.citations.length).toBeGreaterThan(0)
  expect(result.total_seconds).toBeGreaterThan(0)
  expect(result.sources.some((source: { text: string }) => source.text.includes('A database is a collection of related data'))).toBe(true)
  await expect(page.getByText(result.answer, { exact: true })).toBeVisible()
  await page.getByRole('button', { name: `Mở nguồn ${result.citations[0]}`, exact: true }).click()
  await expect(page.getByText(`Đoạn nguồn ${result.citations[0]}`, { exact: true })).toBeVisible()
  await page.screenshot({ path: 'test-results/live-local.png', fullPage: true })
  await import('node:fs/promises').then(fs => fs.writeFile('test-results/live-response.json', JSON.stringify(result, null, 2)))
})
