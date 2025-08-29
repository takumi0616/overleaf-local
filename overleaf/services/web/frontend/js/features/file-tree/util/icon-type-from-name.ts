import { AvailableUnfilledIcon } from '@/shared/components/material-icon'

// TODO ide-redesign-cleanup: Make this the default export and remove the legacy version
export const newEditorIconTypeFromName = (
  name: string
): AvailableUnfilledIcon => {
  let ext = name.split('.').pop()
  ext = ext ? ext.toLowerCase() : ext

  if (ext && ['png', 'pdf', 'jpg', 'jpeg', 'gif'].includes(ext)) {
    return 'image'
  } else if (ext && ['csv', 'xls', 'xlsx'].includes(ext)) {
    return 'table_chart'
  } else if (ext && ['py', 'r'].includes(ext)) {
    return 'code'
  } else if (ext && ['bib'].includes(ext)) {
    return 'book_5'
  }
  return 'description'
}

export default function iconTypeFromName(name: string): string {
  let ext = name.split('.').pop()
  ext = ext ? ext.toLowerCase() : ext

  if (ext && ['png', 'pdf', 'jpg', 'jpeg', 'gif'].includes(ext)) {
    return 'image'
  } else if (ext && ['csv', 'xls', 'xlsx'].includes(ext)) {
    return 'table_chart'
  } else if (ext && ['py', 'r'].includes(ext)) {
    return 'code'
  } else if (ext && ['bib'].includes(ext)) {
    return 'menu_book'
  } else {
    return 'description'
  }
}
