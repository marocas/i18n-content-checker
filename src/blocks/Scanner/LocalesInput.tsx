'use client'

import { useField } from '@payloadcms/ui'
import type { JSONFieldClientProps } from 'payload'
import React, { useCallback, useRef, useState } from 'react'

import './LocalesInput.css'

interface LocaleRow {
  code: string
  flag: string
  label: string
}

export const LocalesInput: React.FC<JSONFieldClientProps> = ({ field, path }) => {
  const { value, setValue } = useField<LocaleRow[]>({ path })
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)
  const dragNode = useRef<HTMLDivElement | null>(null)

  const locales: LocaleRow[] = Array.isArray(value) ? value : []

  const addLocale = useCallback(() => {
    setValue([...locales, { code: '', flag: '', label: '' }])
  }, [locales, setValue])

  const removeLocale = useCallback(
    (index: number) => {
      setValue(locales.filter((_, i) => i !== index))
    },
    [locales, setValue],
  )

  const updateLocale = useCallback(
    (index: number, field: keyof LocaleRow, val: string) => {
      const updated = locales.map((locale, i) =>
        i === index ? { ...locale, [field]: val } : locale,
      )
      setValue(updated)
    },
    [locales, setValue],
  )

  const handleDragStart = useCallback((e: React.DragEvent<HTMLDivElement>, index: number) => {
    setDragIndex(index)
    dragNode.current = e.currentTarget
    e.dataTransfer.effectAllowed = 'move'
  }, [])

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>, index: number) => {
      e.preventDefault()
      if (dragIndex === null || dragIndex === index) return
      setDragOverIndex(index)
    },
    [dragIndex],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>, index: number) => {
      e.preventDefault()
      if (dragIndex === null || dragIndex === index) return
      const reordered = [...locales]
      const [moved] = reordered.splice(dragIndex, 1)
      reordered.splice(index, 0, moved)
      setValue(reordered)
      setDragIndex(null)
      setDragOverIndex(null)
    },
    [dragIndex, locales, setValue],
  )

  const handleDragEnd = useCallback(() => {
    setDragIndex(null)
    setDragOverIndex(null)
  }, [])

  return (
    <div className="locales-input-field">
      <label className="locales-input-label">
        {typeof field?.label === 'string' ? field.label : 'Locales to Scan'}
      </label>
      <div className="locales-input-list">
        {locales.map((locale, index) => (
          <div
            key={index}
            className={`locales-input-row${dragOverIndex === index ? ' locales-input-row--drag-over' : ''}${dragIndex === index ? ' locales-input-row--dragging' : ''}`}
            draggable
            onDragStart={(e) => handleDragStart(e, index)}
            onDragOver={(e) => handleDragOver(e, index)}
            onDrop={(e) => handleDrop(e, index)}
            onDragEnd={handleDragEnd}
          >
            <span className="locales-input-drag-handle" title="Drag to reorder">
              ⠿
            </span>
            <input
              type="text"
              className="locales-input-text locales-input-code"
              value={locale.code}
              onChange={(e) => updateLocale(index, 'code', e.target.value)}
              placeholder="pt-pt"
            />
            <input
              type="text"
              className="locales-input-text locales-input-flag"
              value={locale.flag}
              onChange={(e) => updateLocale(index, 'flag', e.target.value)}
              placeholder="🇵🇹"
            />
            <input
              type="text"
              className="locales-input-text locales-input-label-field"
              value={locale.label}
              onChange={(e) => updateLocale(index, 'label', e.target.value)}
              placeholder="Portuguese (PT)"
            />
            <button
              type="button"
              className="locales-input-remove"
              onClick={() => removeLocale(index)}
              aria-label={`Remove ${locale.label || 'locale'}`}
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <button type="button" className="locales-input-add" onClick={addLocale}>
        + Add Locale
      </button>
    </div>
  )
}
