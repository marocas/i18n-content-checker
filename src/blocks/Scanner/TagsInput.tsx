'use client'

import { useField } from '@payloadcms/ui'
import type { JSONFieldClientProps } from 'payload'
import React, { useCallback, useState } from 'react'

import './TagsInput.css'

export const TagsInput: React.FC<JSONFieldClientProps> = ({ field, path }) => {
  const { value, setValue } = useField<string[]>({ path })
  const [inputValue, setInputValue] = useState('')

  const tags = Array.isArray(value) ? value : []

  const addTag = useCallback(
    (tag: string) => {
      const trimmed = tag.trim()
      if (trimmed && !tags.includes(trimmed)) {
        setValue([...tags, trimmed])
      }
      setInputValue('')
    },
    [tags, setValue],
  )

  const removeTag = useCallback(
    (index: number) => {
      setValue(tags.filter((_, i) => i !== index))
    },
    [tags, setValue],
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault()
        addTag(inputValue)
      }
      if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
        removeTag(tags.length - 1)
      }
    },
    [inputValue, tags, addTag, removeTag],
  )

  return (
    <div className="tags-input-field">
      {field?.label && (
        <label className="tags-input-label">
          {typeof field.label === 'string' ? field.label : 'Excluded Terms'}
        </label>
      )}
      <div className="tags-input-container">
        {tags.map((tag, index) => (
          <span key={index} className="tags-input-chip">
            {tag}
            <button
              type="button"
              className="tags-input-chip-remove"
              onClick={() => removeTag(index)}
              aria-label={`Remove ${tag}`}
            >
              ×
            </button>
          </span>
        ))}
        <input
          type="text"
          className="tags-input-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={tags.length === 0 ? 'Type and press Enter to add...' : 'Add another...'}
        />
      </div>
    </div>
  )
}
